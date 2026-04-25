import os
import sys
import numpy as np
import torch
import mne

import moabb

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from util.Performance_Viewer import PerformanceViewer
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface
from EEG_app.src.components.DataProvider.FifDataProvider import FifDataProvider, LABEL_MAP
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer

from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator

from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.EpochProcessing.BadChannelDetectors.GradientDetector import GradientDetector

moabb.set_log_level("ERROR")
SEED = 42

def run_pipeline(train_provider, test_provider, model_interface, epochs, epoch_pipeline):
    torch.manual_seed(SEED)

    # ── Datos de entrenamiento ──────────────────────────────────────────────
    X_train, Y_train, _ = train_provider.get_data()
    X_val, Y_val, _ = test_provider.get_data() 

    # La primera llamada ajusta la matriz de Euclidean Alignment sobre X_train.
    # Las llamadas posteriores reutilizan esa misma matriz.
    X_train, Y_train = epoch_pipeline.process_np(X_train, Y_train, shuffle=False)  # ajusta EA
    X_val,   Y_val   = epoch_pipeline.process_np(X_val,   Y_val,   shuffle=False)  # aplica EA

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    final_val_acc = model_interface.finetuning(X_train, Y_train, X_val, Y_val, epochs=epochs)

    # ── Predicción y métricas ────────────────────────────────────────────────
    preds_array, probs_array = model_interface.predict_batch(X_val)

    acc  = accuracy_score(Y_val, preds_array)
    prec = precision_score(Y_val, preds_array, average="macro", zero_division=0)
    rec  = recall_score(Y_val,   preds_array, average="macro", zero_division=0)
    f1   = f1_score(Y_val,       preds_array, average="macro", zero_division=0)

    viewer = PerformanceViewer()
    viewer.summary(final_val_acc)
    viewer.plot_fine_tune(final_val_acc)
    viewer.plot_downstream(preds_array, probs_array, Y_val)

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
    }

def run_MiRepNet_pipeline(train_fif_paths, test_fif_paths, annotations_names=["left_hand", "right_hand"], epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_names = FifDataProvider(fif_paths=train_fif_paths).get_channel_names()

    raw_pipeline = RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])

    train_provider = FifDataProvider(
        fif_paths=train_fif_paths,
        raw_pipeline_detection=raw_pipeline,
        annotations_names=annotations_names,
    )
    test_provider = FifDataProvider(
        fif_paths=test_fif_paths,
        raw_pipeline_detection=raw_pipeline,
        annotations_names=annotations_names,
    )

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=annotations_names,
    )

    results = run_pipeline(train_provider, test_provider, modelo, epochs, epoch_pipeline)

    print("\n── Resultados ─────────────────────────────────────────────────")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results


def run_MiRepNet_pipeline_bads(train_fif_paths, test_fif_paths, annotations_names=["left_hand", "right_hand"], epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener nombres de canales antes de construir el interpolador
    channel_names = FifDataProvider(fif_paths=train_fif_paths).get_channel_names()

    bad_channel_interpolator = BadChannelInterpolator(
        channels_max=3,
        print_history=True,
        actual_channel_positions=channel_names,
        detectors=[
            AmplitudeThresholdDetector(threshold=100),
            VarianceDetector(threshold=1000.0, dead_threshold=2),
        ]
    )

    raw_pipeline_detection = RawProcessorPipeline([BandpassFilter(1, 40.0),   AnnotationRenamer(LABEL_MAP)])
    raw_pipeline_final     = RawProcessorPipeline([BandpassFilter(8, 30.0),   AnnotationRenamer(LABEL_MAP)])

    train_provider = FifDataProvider(
        fif_paths=train_fif_paths,
        raw_pipeline_detection=raw_pipeline_detection,
        raw_pipeline_final=raw_pipeline_final,
        bad_channel_interpolator=bad_channel_interpolator,
        annotations_names=annotations_names,
    )

    test_provider = FifDataProvider(
        fif_paths=test_fif_paths,
        raw_pipeline_detection=raw_pipeline_detection,
        raw_pipeline_final=raw_pipeline_final,
        bad_channel_interpolator=bad_channel_interpolator,
        annotations_names=annotations_names,
    )

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=annotations_names,
    )

    results = run_pipeline(train_provider, test_provider, modelo, epochs, epoch_pipeline)

    print("\n── Resultados ─────────────────────────────────────────────────")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results


def main():
    train_fif = [
        "EEG_app/recordings/experimento_visual/suj1/suj1_2_raw.fif",
        "EEG_app/recordings/experimento_visual/suj1/suj1_3_raw.fif",
    ]

    test_fif = [
        "EEG_app/recordings/experimento_visual/suj1/suj1_1_raw.fif",
        "EEG_app/recordings/experimento_visual/suj1/suj1_4_raw.fif",
    ]

    run_MiRepNet_pipeline_bads(
        train_fif_paths=train_fif,
        test_fif_paths=test_fif,
        epochs=10,
    )


if __name__ == "__main__":
    main()
    