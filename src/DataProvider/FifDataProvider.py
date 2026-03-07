from typing import Dict, List, Optional, Sequence

import mne
import numpy as np
import os, sys

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from DataProvider.DataProvider import DataProvider
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
}

CLASS_NAMES = ["feet", "left_hand", "right_hand"]  # orden alfabético = orden real de LabelEncoder

def _raw_to_epochs(raw, tmin=0.0, tmax=4.0):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in CLASS_NAMES}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    return epochs

class FifDataProvider(DataProvider):
    def __init__(self, fif_paths: str | Sequence[str] = [], raw_pipeline: Optional[RawProcessorPipeline] = None) -> None:
        if isinstance(fif_paths, str):
            fif_paths = [fif_paths]
        self._fif_paths = list(fif_paths)
        self._raw_pipeline = raw_pipeline

    def add_fif_path(self, path: str) -> None:
        self._fif_paths.append(path)
    
    def remove_fif_path(self, path: str) -> None:
        self._fif_paths.remove(path)    

    def get_fif_paths(self) -> List[str]:
        return list(self._fif_paths)
    
    def get_data(self):
        all_epochs_list: List[np.ndarray] = []
        all_labels: List[str] = []

        for path in self._fif_paths:
            print(f"Cargando {path} ...")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

            if self._raw_pipeline is not None:
                raw = self._raw_pipeline.process(raw)

            single_epoch = _raw_to_epochs(raw)

            all_epochs_list.append(single_epoch)

        combined_epochs = mne.concatenate_epochs(all_epochs_list)
        
        X = combined_epochs.get_data()

        true_labels_numeric = combined_epochs.events[:, 2]
        inv_event_id = {v: k for k, v in combined_epochs.event_id.items()}
        true_labels = [inv_event_id[i] for i in true_labels_numeric]

        classes = sorted(set(true_labels))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[l] for l in true_labels], dtype=np.int64)

        return X, y, classes
    
    def get_channel_names(self) -> List[str]:
        if not self._fif_paths:
            raise ValueError("No se han proporcionado archivos FIF para obtener los nombres de los canales.")
        
        raw = mne.io.read_raw_fif(self._fif_paths[0], preload=False, verbose=False)
        raw = raw.pick_types(eeg=True)
        
        return raw.ch_names