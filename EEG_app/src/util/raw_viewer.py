import argparse
import mne
import os, sys
import matplotlib.pyplot as plt

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.NotchFilter import NotchFilter
from components.RawProcessing.Resampler import Resampler
from components.RawProcessing.CARReference import CARReference
from components.RawProcessing.ICAProcessor import ICAProcessor
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer

PATH = "C:/Users/julianix/Documents/programacion/python/TFG-BCI/EEG_app/recordings/experimento_visual/suj5/suj5_4_raw.fif"

raw_pipeline = RawProcessorPipeline([
    NotchFilter(50.0),
    BandpassFilter(1, 40.0),
    # AnnotationRenamer(LABEL_MAP),
    # CARReference(),
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