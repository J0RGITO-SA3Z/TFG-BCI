import os
import sys
import torch

import moabb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.Performance_Viewer import PerformanceViewer
from model_interface.MiRepNetInterface import MiRepNetInterface
from DataProvider.FifDataProvider import FifDataProvider

from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.BadChannelInterpolator import BadChannelInterpolator

from epoch_processing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from epoch_processing.BadChannelDetectors.VarianceDetector import VarianceDetector
from epoch_processing.BadChannelDetectors.GradientDetector import GradientDetector

moabb.set_log_level("ERROR")
SEED = 42


def run_pipeline(train_provider, test_provider, model_interface, epochs, epoch_pipeline, validation_split=0.2, show_plots=True):
    torch.manual_seed(SEED)

    # ── Datos de entrenamiento ──────────────────────────────────────────────
    X_train_all, Y_train_all, _ = train_provider.get_data()

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_all, Y_train_all,
        test_size=validation_split,
        random_state=SEED,
        stratify=Y_train_all,
    )

    # La primera llamada ajusta la matriz de Euclidean Alignment sobre X_train.
    # Las llamadas posteriores reutilizan esa misma matriz.
    X_train, Y_train = epoch_pipeline.process_np(X_train, Y_train, shuffle=False)  # ajusta EA
    X_val,   Y_val   = epoch_pipeline.process_np(X_val,   Y_val,   shuffle=False)  # aplica EA

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    final_val_acc = model_interface.finetuning(X_train, Y_train, X_val, Y_val, epochs=epochs)

    # ── Datos de test ────────────────────────────────────────────────────────
    X_test, Y_test, _ = test_provider.get_data()
    X_test, Y_test    = epoch_pipeline.process_np(X_test, Y_test, shuffle=False)  # aplica EA

    # ── Predicción y métricas ────────────────────────────────────────────────
    preds_array, probs_array = model_interface.predict_batch(X_test)

    acc  = accuracy_score(Y_test, preds_array)
    prec = precision_score(Y_test, preds_array, average="macro", zero_division=0)
    rec  = recall_score(Y_test,   preds_array, average="macro", zero_division=0)
    f1   = f1_score(Y_test,       preds_array, average="macro", zero_division=0)

    if show_plots:
        viewer = PerformanceViewer()
        viewer.summary(final_val_acc)
        viewer.plot_fine_tune(final_val_acc)
        viewer.plot_downstream(preds_array, probs_array, Y_test)

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
    }


def run_MiRepNet_pipeline(train_fif_paths, test_fif_paths, annotations_names=["left_hand", "right_hand","rest"], epochs=10, validation_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_provider = FifDataProvider(fif_paths=train_fif_paths, annotations_names=annotations_names)
    test_provider  = FifDataProvider(fif_paths=test_fif_paths,  annotations_names=annotations_names)

    epoch_pipeline = EpochProcessorPipeline([
        BadChannelInterpolator(
            channels_max=4,
            print_history=True,
            actual_channel_positions=train_provider.get_channel_names(),
            detectors=[
                AmplitudeThresholdDetector(threshold=100),
                VarianceDetector(threshold=1000.0, dead_threshold=1e-10),
                GradientDetector(threshold=25.0),
            ],
        ),
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=train_provider.get_channel_names()),
    ])

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=annotations_names,
    )

    results = run_pipeline(train_provider, test_provider, modelo, epochs, epoch_pipeline, validation_split)

    print("\n── Resultados ─────────────────────────────────────────────────")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results


def main():
    train_fif = [
        "EEG_controller_app/recordings/suj2_1_raw.fif",
        "EEG_controller_app/recordings/suj2_2_raw.fif",
        "EEG_controller_app/recordings/suj2_3_raw.fif",
    ]

    test_fif = [
        "EEG_controller_app/recordings/suj2_4_raw.fif",
    ]

    run_MiRepNet_pipeline(
        train_fif_paths=train_fif,
        test_fif_paths=test_fif,
        epochs=10,
        validation_split=0.2,
    )


if __name__ == "__main__":
    main()
    