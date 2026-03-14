import argparse
import mne
import os, sys
import matplotlib.pyplot as plt

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler
from raw_processing.CARReference import CARReference
from raw_processing.ICAProcessor import ICAProcessor
from raw_processing.AnnotationRenamer import AnnotationRenamer

PATH = "C:/Users/julianix/Documents/programacion/python/TFG-BCI/EEG_controller_app/recordings/suj2_1_raw.fif"

raw_pipeline = RawProcessorPipeline([
    NotchFilter(50.0),
    BandpassFilter(8.0, 30.0),
    #AnnotationRenamer(LABEL_MAP),
    #CARReference(),
    # Resampler(250),
    # ICAProcessor(),
])

def main():
    # Cargar raw
    print("Cargando archivo...")
    raw = mne.io.read_raw_fif(PATH, preload=True)

    print("\nMostrando señal SIN filtrar")
    raw.plot(scalings='auto', verbose=False)

    print("\nAplicando pipeline de procesado...")
    processed_raw = raw_pipeline.process(raw)

    print("\nMostrando señal FILTRADA")
    processed_raw.plot(scalings='auto', verbose=False)

    plt.show()

if __name__ == "__main__":
    main()