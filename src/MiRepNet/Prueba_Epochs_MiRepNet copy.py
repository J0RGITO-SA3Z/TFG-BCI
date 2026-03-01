"""
Script para evaluar el modelo MIRepNet con datos EEG personalizados en formato (B, C, T)
con 45 canales de EEG.

Usa ProcessorPipeline del paquete raw_processing para el preprocesado.
"""

import os
import sys

import numpy as np
import mne
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", message=".*frozen.*", module="pydantic")
warnings.filterwarnings("ignore", message=".*repr.*", module="pydantic")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from model_interface.MiRepNetInterface import MiRepNetInterface
from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding

from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler
from raw_processing.CARReference import CARReference
from raw_processing.ICAProcessor import ICAProcessor
from raw_processing.AnnotationRenamer import AnnotationRenamer

from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.EpochNormalizer import EpochNormalizer
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment

# Etiquetas del experimento → etiquetas del modelo
LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
}

CLASS_NAMES = ["feet", "left_hand", "right_hand"]  # orden alfabético = orden real de LabelEncoder

# Pipeline de preprocesamiento 
raw_pipeline = RawProcessorPipeline([
    BandpassFilter(8.0, 30.0),
    NotchFilter(50.0),
    Resampler(250),
    CARReference(),
    ICAProcessor(),
    AnnotationRenamer(LABEL_MAP),
])

# Pipeline de procesamiento sobre epochs (se ejecuta después de epoquizar)
epoch_pipeline = EpochProcessorPipeline([
    SpatialInterpolator(),        # interpola/reordena canales a la topología objetivo
    EuclideanAlignment(),         # alineamiento euclídeo (EA)
    EpochNormalizer(),            # normalización z-score por epoch
])

# Función para convertir un Raw (ya preprocesado por el pipeline) a epochs (B, C, T)
def raw_to_epochs(raw, tmin=0.0, tmax=4.0):
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

def epoch_to_numpy(epochs):
    """
    Convierte un mne.Epochs a un array numpy (B, C, T) y las etiquetas correspondientes.
    Asume que las etiquetas ya están codificadas como enteros (0, 1, 2) en el orden de CLASS_NAMES.
    """

    true_labels_numeric = epochs.events[:, 2]
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    true_labels = [inv_event_id[i] for i in true_labels_numeric]

    return epochs.get_data(), true_labels

def experimento(fineTuneFile, validationFile, epochs=10):
    # === Configuración del Dispositivo ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar modelo preentrenado
    model = MiRepNetInterface(weight_path=WEIGHT_PATH)

    # Cargar datos raw
    finetuneRaw = mne.io.read_raw_fif(fineTuneFile, preload=True)
    valRaw = mne.io.read_raw_fif(validationFile, preload=True)

    #aplica preprocesado al raw
    finetuneRaw = raw_pipeline.process(finetuneRaw)
    valRaw      = raw_pipeline.process(valRaw)

    #separamos los raws en eopchs
    epochs_finetune = raw_to_epochs(finetuneRaw)
    epochs_val = raw_to_epochs(valRaw)

    #aplica el procesamiento sobre los epochs
    processed_epochs_finetune = epoch_pipeline.process(epochs_finetune)
    processed_epochs_val = epoch_pipeline.process(epochs_val)

    #convertimos los epochs a numpy arrays (B, C, T) y etiquetas correspondientes
    X_finetune, y_finetune = epoch_to_numpy(processed_epochs_finetune)
    X_val, y_val = epoch_to_numpy(processed_epochs_val)

    # fine-tunea el modelo con los epochs de finetune y evalúa con los epochs de validación
    model.finetuning(X_finetune, y_finetune, epochs=epochs, valData=X_val, valLabels=y_val)
    res = model.predict_batch(X_val)


def main():
    fineTuneFile = os.path.join(PROJECT_ROOT,"..", "EEG_controller_app","recordings", "suj2_1.fif")
    validationFile = os.path.join(PROJECT_ROOT,"..", "EEG_controller_app","recordings", "suj2_2.fif")

    experimento(fineTuneFile, validationFile, epochs=10)


if __name__ == "__main__":
    main()