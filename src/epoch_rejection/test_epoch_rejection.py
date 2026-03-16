import os
import sys
import numpy as np
import mne

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ─────── Data Providers ─────────────────────────────────────────────────────
from DataProvider.FifDataProvider import FifDataProvider

# ─────── FASTER (Fully Automated Statical Thresholding EEG Rejection) ─────────────────────────────────────────────────────
from mne_faster import (
    find_bad_channels,
    find_bad_channels_in_epochs,
    find_bad_components,
    find_bad_epochs,
)

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler
from raw_processing.CARReference import CARReference
from raw_processing.ICAProcessor import ICAProcessor
from raw_processing.AnnotationRenamer import AnnotationRenamer

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

def _raw_to_epochs(raw, tmin=0.0, tmax=4.0, anotationsNames=["left_hand", "right_hand", "feet"]):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in anotationsNames}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    return epochs


def test_faster(path,anotationsNames=["right_hand"]):

    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)

    _raw_pipeline = RawProcessorPipeline([
                # NotchFilter(50.0),
                BandpassFilter(8.0, 30.0),
                AnnotationRenamer(LABEL_MAP),
                #CARReference(),
                # Resampler(250),
                # ICAProcessor(),
            ])
    
    raw = _raw_pipeline.process(raw)
    epoch = _raw_to_epochs(raw, anotationsNames = anotationsNames)

    # Compute evoked before cleaning, using an average EEG reference
    epoch_before = epoch.copy()
    epoch_before.set_eeg_reference("average")
    evoked_before = epoch_before.average()

    # Aplicamos FASTER a los epochs procesados por nuestro pipeline
    # No se puede palicar FASTER directamente porque no tenemos canales eog 
    # cleaned = run_faster(single_epoch, thres=3.0, copy=True)

    for ch in epoch.info["chs"]:
        print(ch["ch_name"], ch["loc"][:3])

    # Step 1: mark bad channels
    # montage = mne.channels.make_standard_montage("standard_1020")
    # epoch.set_montage(montage, on_missing="ignore")

    epoch.info["bads"] = find_bad_channels(epoch, eeg_ref_corr=False)
    
    # Step 2: bad epochs según FASTER
    bad_epoch_idx = find_bad_epochs(epoch)  # devuelve lista de índices

    print(f"FASTER bad channels : {epoch.info['bads']}")
    print(f"FASTER bad epochs   : {bad_epoch_idx}")
    print(f"Nº de epochs antes  : {len(epoch)}")

    # Interpolar canales malos (opcional pero recomendado antes de drop epochs)
    if epoch.info["bads"]:
        epoch.interpolate_bads(reset_bads=True)

    # Descartar epochs malos
    epoch.drop(bad_epoch_idx, reason="FASTER")

    print(f"Nº de epochs después: {len(epoch)}")


    # Compute evoked after cleaning, using an average EEG reference
    epoch.set_eeg_reference("average")
    evoked_after = epoch.average()

    # Plot the evokeds of the data, before and after cleaning
    evoked_before.plot()
    evoked_after.plot()

def main() -> None:
    
    fif_path = "EEG_controller_app/recordings/suj2_1_raw.fif"

    test_faster(
        path    = fif_path,
    )


if __name__ == "__main__":
    main()
