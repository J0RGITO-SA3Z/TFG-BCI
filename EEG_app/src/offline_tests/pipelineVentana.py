"""
Pipeline ventana: pretraining sobre raw completo + evaluación con ventana deslizante.

Pretraining:
    - Carga archivos FIF de entrenamiento.
    - Aplica el filtro de banda sobre todo el raw.
    - Epoquiza y hace fine-tuning del modelo.

Evaluación (ventana):
    - Carga archivos FIF de evaluación.
    - Para cada evento, recorta una ventana de ``window_size`` segundos.
    - Aplica el filtro de banda sólo sobre esa ventana.
    - Toma los últimos ``epoch_duration`` segundos como epoch.
    - Evalúa el modelo ya entrenado sobre esos epochs.
"""

import os
import sys
import torch

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
from components.DataProvider.FifVentanaDataProvider import FifVentanaDataProvider

from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.ClassEventRemover import ClassEventRemover
from components.EpochProcessing.EpochEventRenamer import EpochEventRenamer
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator

from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.EpochProcessing.BadChannelDetectors.GradientDetector import GradientDetector
from components.Trainings.trainingOffline import Training_offline
import matplotlib.pyplot as plt

moabb.set_log_level("ERROR")
SEED = 42

def run_MiRepNet_ventana_pipeline(
    train_fif_paths,
    eval_fif_paths,
    annotations_names=["left_hand", "right_hand"],
    window_size=40.0,
    epoch_duration=4.0,
    epochs=10,
    validation_split=0.2,
    show_plots=True,
):
    """
    Punto de entrada de alto nivel para el pipeline ventana con MIRepNet.

    El entrenamiento replica exactamente Training_offline (usado en Main_pipeline_RT.py):
    calcula la matriz EA sobre todos los datos de entrenamiento y hace fine-tuning.

    La evaluación aplica esa matriz EA fija (igual que RT_pipeline_process.py), de modo
    que el alineamiento euclídeo en inferencia es idéntico al del pipeline en tiempo real.

    Args:
        train_fif_paths:   Lista de rutas .fif para entrenamiento.
        eval_fif_paths:    Lista de rutas .fif para evaluación (preprocesado por ventana).
        annotations_names: Nombres de las clases a predecir.
        window_size:       Segundos de ventana a recortar desde el onset del evento.
        epoch_duration:    Segundos a tomar del final de la ventana como epoch.
        epochs:            Épocas de fine-tuning.
        validation_split:  Fracción de validación interna del fine-tuning.
        show_plots:        Si True, muestra gráficas de fine-tuning y downstream.
    """
    torch.manual_seed(SEED)

    # ── Entrenamiento offline (idéntico a Main_pipeline_RT.py) ─────────────
    rt_training = Training_offline()
    EA_matrix, model = rt_training.start(
        train_fif_paths,
        lista=annotations_names,
        epochs=epochs,
        seed=SEED,
        validation_split=validation_split,
    )

    # ── Evaluación con matriz EA fija (idéntico a RT_pipeline_process.py) ──
    eval_provider = FifVentanaDataProvider(
        fif_paths=eval_fif_paths,
        annotations_names=annotations_names,
        window_size=window_size,
        epoch_duration=epoch_duration,
    )

    eval_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(matrix=EA_matrix),
        SpatialInterpolator(actual_channel_positions=rt_training.getChannelNames()),
    ])

    X_eval, Y_eval, _ = eval_provider.get_data()
    X_eval, Y_eval    = eval_pipeline.process_np(X_eval, Y_eval, shuffle=False)

    preds_array, probs_array = model.predict_batch(X_eval)

    acc  = accuracy_score(Y_eval, preds_array)
    prec = precision_score(Y_eval, preds_array, average="macro", zero_division=0)
    rec  = recall_score(Y_eval,  preds_array, average="macro", zero_division=0)
    f1   = f1_score(Y_eval,     preds_array, average="macro", zero_division=0)

    if show_plots:
        viewer = PerformanceViewer()
        viewer.plot_fine_tune(rt_training.getHistory())
        viewer.plot_downstream(preds_array, probs_array, Y_eval)

    print("\n── Resultados evaluación ventana ──────────────────────────────")
    for metric, value in {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}.items():
        print(f"  {metric}: {value:.4f}")

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
    }


def main():

    train_fif = [
        "EEG_controller_app/recordings/suj2_1_raw.fif",
        "EEG_controller_app/recordings/suj2_2_raw.fif",
        "EEG_controller_app/recordings/suj2_3_raw.fif",
    ]

    eval_fif = [
        "EEG_controller_app/recordings/suj2_4_raw.fif",
    ]

    EPOCH_DURATION = 4.0
    window_sizes   = list(range(4, 51))   # 4, 5, 6, ..., 50

    metrics_by_window = {w: None for w in window_sizes}

    for w in window_sizes:
        print(f"\n{'='*55}")
        print(f"  Ventana: {w}s")
        print(f"{'='*55}")
        results = run_MiRepNet_ventana_pipeline(
            train_fif_paths=train_fif,
            eval_fif_paths=eval_fif,
            annotations_names=["left_hand", "right_hand","rest"],
            window_size=float(w),
            epoch_duration=EPOCH_DURATION,
            epochs=10,
            validation_split=0.2,
            show_plots=False,
        )
        metrics_by_window[w] = results

    # ── Construir series ─────────────────────────────────────────────────────
    x      = window_sizes
    acc    = [metrics_by_window[w]["Accuracy"]  for w in x]
    prec   = [metrics_by_window[w]["Precision"] for w in x]
    rec    = [metrics_by_window[w]["Recall"]    for w in x]
    f1     = [metrics_by_window[w]["F1-Score"]  for w in x]

    # ── Gráfico ──────────────────────────────────────────────────────────────
    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, acc,  marker="o", linewidth=1.8, markersize=4, label="Accuracy")
    ax.plot(x, prec, marker="s", linewidth=1.8, markersize=4, label="Precision")
    ax.plot(x, rec,  marker="^", linewidth=1.8, markersize=4, label="Recall")
    ax.plot(x, f1,   marker="D", linewidth=1.8, markersize=4, label="F1-Score")

    ax.axvline(x=EPOCH_DURATION, color="gray", linestyle="--", linewidth=1,
               label=f"epoch_duration ({int(EPOCH_DURATION)}s)")

    ax.set_xlabel("Tamaño de ventana (s)")
    ax.set_ylabel("Métrica")
    ax.set_title("Efecto del tamaño de ventana en las métricas de clasificación")
    ax.set_xticks(range(4, 51, 2))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main2():
    train_fif = [
        "EEG_controller_app/recordings/suj2_1_raw.fif",
        "EEG_controller_app/recordings/suj2_2_raw.fif",
        "EEG_controller_app/recordings/suj2_3_raw.fif",
    ]

    eval_fif = [
        "EEG_controller_app/recordings/suj2_4_raw.fif",
    ]

    run_MiRepNet_ventana_pipeline(
        train_fif_paths=train_fif,
        eval_fif_paths=eval_fif,
        annotations_names=["left_hand", "right_hand","rest"],
        window_size=15.0,
        epoch_duration=4.0,
        epochs=10,
        validation_split=0.2,
        show_plots=True,
    )


if __name__ == "__main__":
    print("Selecciona qué ejecutar:")
    print("  1 - Barrido de ventanas con gráfica (main)")
    print("  2 - Ejecución única con ventana fija  (main2)")
    opcion = input("Opción [1/2]: ").strip()

    if opcion == "1":
        main()
    elif opcion == "2":
        main2()
    else:
        print(f"Opción '{opcion}' no reconocida. Usa 1 o 2.")
