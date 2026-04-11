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

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.Performance_Viewer import PerformanceViewer
from model_interface.MiRepNetInterface import MiRepNetInterface
from DataProvider.FifDataProvider import FifDataProvider
from DataProvider.FifVentanaDataProvider import FifVentanaDataProvider

from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.ClassEventRemover import ClassEventRemover
from epoch_processing.EpochEventRenamer import EpochEventRenamer
from epoch_processing.BadChannelInterpolator import BadChannelInterpolator

from epoch_processing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from epoch_processing.BadChannelDetectors.VarianceDetector import VarianceDetector
from epoch_processing.BadChannelDetectors.GradientDetector import GradientDetector
import matplotlib.pyplot as plt

moabb.set_log_level("ERROR")
SEED = 42

def run_ventana_pipeline(
    train_provider,
    eval_provider,
    model_interface,
    epochs,
    epoch_pipeline,
    validation_split=0.2,
    exclude_training_classes=None,
    rename_training_classes=None,
    show_plots=True,
):
    """
    Ejecuta el pipeline ventana:
      - Entrena/fine-tunea con los datos de ``train_provider``.
      - Evalúa sobre los epochs generados por ``eval_provider`` (ventana deslizante).

    Args:
        train_provider:           DataProvider con los datos de entrenamiento
                                  (preprocesado sobre raw completo).
        eval_provider:            FifVentanaDataProvider con los datos de evaluación
                                  (preprocesado sólo sobre la ventana).
        model_interface:          Interfaz del modelo (fine-tuning + predict).
        epochs:                   Número de épocas de fine-tuning.
        epoch_pipeline:           Pipeline de procesado a nivel de epoch
                                  (EuclideanAlignment, SpatialInterpolator, etc.).
        validation_split:         Fracción de los datos de entrenamiento usada
                                  como validación interna del fine-tuning.
        exclude_training_classes: Clases a excluir del entrenamiento (ej. ["rest"]).
        rename_training_classes:  Renombrado de clases {original: nuevo}.
        show_plots:               Si True, muestra gráficas de rendimiento.

    Returns:
        dict con Accuracy, Precision, Recall y F1-Score sobre los datos de evaluación.
    """
    torch.manual_seed(SEED)

    # ── Datos de entrenamiento ──────────────────────────────────────────────
    X_train_all, Y_train_all, _ = train_provider.get_data()

    X_train, X_val_int, Y_train, Y_val_int = train_test_split(
        X_train_all, Y_train_all,
        test_size=validation_split,
        random_state=SEED,
        stratify=Y_train_all,
    )

    X_val_int2, Y_val_int2 = X_val_int, Y_val_int

    if exclude_training_classes is not None:
        remover = ClassEventRemover(exclude_training_classes)
        X_train,    Y_train    = remover.process_np(X_train,    Y_train)
        X_val_int2, Y_val_int2 = remover.process_np(X_val_int,  Y_val_int)

    if rename_training_classes is not None:
        renamer = EpochEventRenamer(rename_training_classes)
        X_train,    Y_train    = renamer.process_np(X_train,    Y_train)
        X_val_int2, Y_val_int2 = renamer.process_np(X_val_int2, Y_val_int2)

    # La primera llamada (X_train) calcula y guarda la matriz de Euclidean Alignment.
    # Todas las llamadas posteriores (val interno y eval ventana) reutilizan esa
    # misma matriz, de modo que el alineamiento siempre se hace respecto a la
    # distribución de covarianza del conjunto de entrenamiento.
    X_train,    Y_train    = epoch_pipeline.process_np(X_train,    Y_train,    shuffle=False)  # ajusta EA
    X_val_int,  Y_val_int  = epoch_pipeline.process_np(X_val_int,  Y_val_int,  shuffle=False)  # aplica EA
    X_val_int2, Y_val_int2 = epoch_pipeline.process_np(X_val_int2, Y_val_int2, shuffle=False)  # aplica EA

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    final_val_acc = model_interface.finetuning(X_train, Y_train, X_val_int2, Y_val_int2, epochs=epochs)

    # ── Datos de evaluación (ventana) ────────────────────────────────────────
    # La matriz EA ya está fijada: se aplica la del entrenamiento.
    X_eval, Y_eval, _ = eval_provider.get_data()
    X_eval, Y_eval    = epoch_pipeline.process_np(X_eval, Y_eval, shuffle=False)  # aplica EA

    # ── Predicción y métricas ────────────────────────────────────────────────
    preds_array, probs_array = model_interface.predict_batch(X_eval)

    acc  = accuracy_score(Y_eval, preds_array)
    prec = precision_score(Y_eval, preds_array, average="macro", zero_division=0)
    rec  = recall_score(Y_eval,  preds_array, average="macro", zero_division=0)
    f1   = f1_score(Y_eval,     preds_array, average="macro", zero_division=0)

    if show_plots:
        viewer = PerformanceViewer()
        viewer.summary(final_val_acc)
        viewer.plot_fine_tune(final_val_acc)
        viewer.plot_downstream(preds_array, probs_array, Y_eval)

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1,
    }


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

    Args:
        train_fif_paths:   Lista de rutas .fif para entrenamiento (preprocesado completo).
        eval_fif_paths:    Lista de rutas .fif para evaluación (preprocesado por ventana).
        annotations_names: Nombres de las clases a predecir.
        window_size:       Segundos de ventana a recortar desde el onset del evento.
        epoch_duration:    Segundos a tomar del final de la ventana como epoch.
        epochs:            Épocas de fine-tuning.
        validation_split:  Fracción de validación interna del fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_provider = FifDataProvider(
        fif_paths=train_fif_paths,
        annotations_names=annotations_names,
    )

    eval_provider = FifVentanaDataProvider(
        fif_paths=eval_fif_paths,
        annotations_names=annotations_names,
        window_size=window_size,
        epoch_duration=epoch_duration,
    )

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=train_provider.get_channel_names()),
    ])

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=annotations_names,
    )

    results = run_ventana_pipeline(
        train_provider=train_provider,
        eval_provider=eval_provider,
        model_interface=modelo,
        epochs=epochs,
        epoch_pipeline=epoch_pipeline,
        validation_split=validation_split,
        show_plots=show_plots,
    )

    print("\n── Resultados evaluación ventana ──────────────────────────────")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results


def main():

    train_fif = [
        "EEG_controller_app/recordings/suj2_1_raw.fif",
        "EEG_controller_app/recordings/suj2_2_raw.fif",
        "EEG_controller_app/recordings/suj2_3_raw.fif",
        "EEG_controller_app/recordings/suj2_4_raw.fif",
    ]

    eval_fif = [
        "EEG_controller_app/recordings/suj2_5_raw.fif",
        "EEG_controller_app/recordings/suj2_6_raw.fif",
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
            annotations_names=["left_hand", "right_hand"],
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
    ]

    eval_fif = [
        "EEG_controller_app/recordings/suj2_2_raw.fif",
    ]

    run_MiRepNet_ventana_pipeline(
        train_fif_paths=train_fif,
        eval_fif_paths=eval_fif,
        annotations_names=["left_hand", "right_hand"],
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
