from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from pipeline_RT.triningOffline import Training_offline
from pipeline_RT.raw_processing.RawProcessorPipeline import RawProcessorPipeline
from pipeline_RT.raw_processing.BandpassFilter import BandpassFilter
from pipeline_RT.epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from pipeline_RT.epoch_processing.EuclideanAlignment import EuclideanAlignment
from pipeline_RT.epoch_processing.SpatialInterpolator import SpatialInterpolator
from pipeline_RT.raw_processing.AnnotationRenamer import AnnotationRenamer
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class OfflinePrediction:
    prediction: str
    last_sample: int
    probs: dict
    true_label: str | None = None
    is_correct: bool | None = None


LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA": "right_hand",
    "ABAJO": "feet",
    "DESCANSO": "rest",
}

# Devuelve un array que indica para cada indice de muestre de MNE su numero de sample correspondiente. 
def _build_sample_track(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Devuelve un vector de referencia temporal por muestra.

    - Si existe canal 'Sample', usa sus valores.
    - Si no existe, usa indices absolutos (first_samp + idx).
    """
    if "Sample" in raw.ch_names:
        sample_track = raw.get_data(picks=["Sample"])[0]
        return np.asarray(sample_track, dtype=np.int64)

    return np.arange(raw.first_samp, raw.first_samp + raw.n_times, dtype=np.int64)

# Obitne una lista de todos los eventos y los devuelve en una tupla con su numero de sample respectivo. 
def _extract_events_by_sample(raw: mne.io.BaseRaw, sample_track: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if len(raw.annotations) == 0:
        return np.array([], dtype=np.int64), []

    # Indices relativos al Raw para cada anotacion.
    event_idx = raw.time_as_index(raw.annotations.onset, use_rounding=True)
    event_idx = np.asarray(event_idx, dtype=np.int64)
    event_idx = np.clip(event_idx, 0, len(sample_track) - 1)

    # Alinea cada anotacion al valor real del canal Sample en ese instante.
    event_samples = sample_track[event_idx]
    event_desc = list(raw.annotations.description)
    return event_samples, event_desc

def _plot_predictions_vs_events(
    predictions: list[OfflinePrediction],
    sliding_predictions: list[OfflinePrediction],
    event_samples: np.ndarray,
    event_desc: list[str],
    sfreq: float,
    class_order: Sequence[str] | None = None,
    window_seconds: float = 30.0,
):
    if not predictions and not sliding_predictions:
        print("No hay predicciones para mostrar.")
        return

    event_pred_samples = np.array([p.last_sample for p in predictions], dtype=np.int64)
    event_pred_labels = [p.prediction for p in predictions]
    sliding_pred_samples = np.array([p.last_sample for p in sliding_predictions], dtype=np.int64)
    sliding_pred_labels = [p.prediction for p in sliding_predictions]

    pred_labels = event_pred_labels + sliding_pred_labels

    if class_order:
        # Mantiene los valores originales de etiqueta en el eje Y.
        classes = list(dict.fromkeys(class_order))
        for label in pred_labels:
            if label not in classes:
                classes.append(label)
    else:
        classes = sorted(set(pred_labels))

    class_to_y = {c: i for i, c in enumerate(classes)}
    event_pred_y = np.array([class_to_y[c] for c in event_pred_labels], dtype=float)
    sliding_pred_y = np.array([class_to_y[c] for c in sliding_pred_labels], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))
    plt.subplots_adjust(bottom=0.25)

    scatter_sliding = None
    if sliding_predictions:
        scatter_sliding = ax.scatter(
            sliding_pred_samples,
            sliding_pred_y,
            s=14,
            alpha=0.65,
            color="tab:blue",
            label="Prediccion ventana deslizante",
            zorder=2,
        )

    scatter_correct = None
    scatter_wrong = None
    if predictions:
        correctness = np.array([bool(p.is_correct) for p in predictions], dtype=bool)
        correct_mask = correctness
        wrong_mask = ~correctness

        scatter_correct = ax.scatter(
            event_pred_samples[correct_mask],
            event_pred_y[correct_mask],
            s=30,
            alpha=0.9,
            color="tab:green",
            label="Prediccion correcta",
            zorder=4,
        )

        scatter_wrong = ax.scatter(
            event_pred_samples[wrong_mask],
            event_pred_y[wrong_mask],
            s=30,
            alpha=0.9,
            color="tab:red",
            label="Prediccion incorrecta",
            zorder=4,
        )

    unique_event_types = list(dict.fromkeys(event_desc))
    cmap = plt.get_cmap("tab20")
    event_color_map = {
        ev: cmap(i % cmap.N)
        for i, ev in enumerate(unique_event_types)
    }

    event_lines = []
    event_texts = []
    for sample, desc in zip(event_samples, event_desc):
        event_color = event_color_map.get(desc, "tab:red")
        line = ax.axvline(sample, color=event_color, alpha=0.45, linewidth=1.2)
        text = ax.text(
            sample,
            len(classes) - 0.15,
            desc,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=event_color,
            alpha=0.8,
        )
        event_lines.append(line)
        event_texts.append(text)

    legend_handles = []
    if scatter_sliding is not None:
        legend_handles.append(scatter_sliding)
    if scatter_correct is not None:
        legend_handles.append(scatter_correct)
    if scatter_wrong is not None:
        legend_handles.append(scatter_wrong)

    for ev in unique_event_types:
        legend_handles.append(
            Line2D([0], [0], color=event_color_map[ev], lw=2, label=f"Evento: {ev}")
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Numero de muestra")
    ax.set_ylabel("Clase predicha")
    ax.set_title("Predicciones offline por eventos (ventanas de 4s) vs eventos del testFif")
    ax.grid(alpha=0.25)

    plotted_samples = []
    if event_pred_samples.size:
        plotted_samples.append(event_pred_samples)
    if sliding_pred_samples.size:
        plotted_samples.append(sliding_pred_samples)
    if event_samples.size:
        plotted_samples.append(event_samples)

    if not plotted_samples:
        print("No hay muestras para representar.")
        return

    all_samples = np.concatenate(plotted_samples)
    full_min = int(all_samples.min())
    full_max = int(all_samples.max())

    visible_span = int(window_seconds * sfreq)
    if visible_span <= 0:
        visible_span = int(30 * sfreq)

    if full_max - full_min <= visible_span:
        ax.set_xlim(full_min, full_max)
    else:
        ax.set_xlim(full_min, full_min + visible_span)

    slider_ax = fig.add_axes([0.12, 0.08, 0.76, 0.05])
    slider = Slider(
        ax=slider_ax,
        label="Inicio ventana (muestra)",
        valmin=float(full_min),
        valmax=float(max(full_min, full_max - visible_span)),
        valinit=float(full_min),
        valstep=1.0,
    )

    def on_slider_change(val):
        start = int(val)
        ax.set_xlim(start, start + visible_span)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    plt.show()

def extract_epochs_from_events(
    raw: mne.io.BaseRaw,
    valid_labels: Sequence[str],
    epoch_seconds: float,
):
    sfreq = float(raw.info["sfreq"])
    epoch_samples = int(round(epoch_seconds * sfreq)) + 1  # tmax inclusivo

    eeg_data = raw.get_data()

    events, event_id = mne.events_from_annotations(raw)

    valid_ids = {k: v for k, v in event_id.items() if k in set(valid_labels)}
    valid_event_values = set(valid_ids.values())
    inv_event_id = {v: k for k, v in valid_ids.items()}

    X_epochs = []
    y_labels = []
    event_starts_abs = []

    for sample_abs, _, event_code in events:
        if event_code not in valid_event_values:
            continue

        start = int(sample_abs - raw.first_samp)
        end = start + epoch_samples

        if start < 0 or end > eeg_data.shape[1]:
            continue

        window = eeg_data[:, start:end]

        X_epochs.append(window)
        y_labels.append(inv_event_id[event_code])
        event_starts_abs.append(sample_abs)

    if len(X_epochs) == 0:
        return None, None, None

    return (
        np.array(X_epochs),
        np.array(y_labels),
        np.array(event_starts_abs),
    )

def extract_sliding_windows_batch(
    raw: mne.io.BaseRaw,
    window_seconds: float,
    step_seconds: float = 0.15,
):
    sfreq = float(raw.info["sfreq"])

    window_samples = int(round(window_seconds * sfreq))
    step_samples = int(round(step_seconds * sfreq))

    eeg_data = raw.get_data()  # shape: [n_channels, n_times]
    n_channels, n_times = eeg_data.shape

    if n_times < window_samples:
        return None, None

    # 🔴 generar TODAS las ventanas posibles (stride 1)
    windows = sliding_window_view(eeg_data, window_shape=window_samples, axis=1)
    # shape: [n_channels, n_windows_full, window_samples]

    # 🔴 aplicar salto (step)
    windows = windows[:, ::step_samples, :]

    # 🔴 reorganizar a formato modelo
    X = np.transpose(windows, (1, 0, 2))  # [n_windows, n_channels, n_times]

    # 🔴 calcular samples absolutos
    n_windows = X.shape[0]
    window_starts_rel = np.arange(0, n_windows * step_samples, step_samples)
    window_starts_abs = window_starts_rel + raw.first_samp

    return X, window_starts_abs

def experimentoOffline(
    pretrainingFif: str | Sequence[str],
    testFif: str,
    lista = ["left_hand", "right_hand"],
    epochs: int = 10,
    seed: int = 42,
    epoch_seconds: float = 4.0,
    sliding_step_seconds: float = 0.15,
):
    print("[Offline] Entrenando modelo con pretrainingFif...")

    rtTraining = Training_offline()
    ea_matrix, model = rtTraining.start(
        fif_paths=pretrainingFif,
        lista=lista,
        epochs=epochs,
        seed=seed,
    )

    print("[Offline] Cargando y preprocesando testFif...")
    raw_original = mne.io.read_raw_fif(testFif, preload=True, verbose=False)

    raw_pipeline = RawProcessorPipeline([
        BandpassFilter(8.0, 30.0),
        AnnotationRenamer(LABEL_MAP),
    ])

    raw_proc = raw_pipeline.process(raw_original)
    sample_track = _build_sample_track(raw_proc)

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(matrix=ea_matrix),
        SpatialInterpolator(actual_channel_positions=raw_proc.ch_names),  # 🔴 sin upper()
    ])

    print("[Offline] Extrayendo epochs manualmente...")

    X_epochs, y_true, event_starts_abs = extract_epochs_from_events(
        raw_proc,
        valid_labels=lista,
        epoch_seconds=epoch_seconds
    )

    predictions: list[OfflinePrediction] = []
    sliding_predictions: list[OfflinePrediction] = []

    if X_epochs is None:
        print("[Offline] No se han encontrado epochs validos en el testFif para predecir.")
    else:
        sfreq = float(raw_proc.info["sfreq"])

        # 🔴 pipeline de epochs
        X_proc, _ = epoch_pipeline.process_np(
            X_epochs,
            np.zeros(len(X_epochs)),
            shuffle=False
        )

        pred_labels, prob_dicts = model.predict_batch(X_proc)

        n_times_epoch = int(X_epochs.shape[2])

        event_start_rel = np.clip(
            event_starts_abs - int(raw_proc.first_samp),
            0,
            len(sample_track) - 1
        )

        event_end_rel = np.clip(
            event_start_rel + n_times_epoch - 1,
            0,
            len(sample_track) - 1
        )

        for pred_label, prob_dict, true_label, end_rel_idx in zip(
            pred_labels, prob_dicts, y_true, event_end_rel
        ):
            predictions.append(
                OfflinePrediction(
                    prediction=str(pred_label),
                    last_sample=int(sample_track[int(end_rel_idx)]),
                    probs=prob_dict,
                    true_label=str(true_label),
                    is_correct=(str(pred_label) == str(true_label)),
                )
            )

    X_sliding, window_starts_abs = extract_sliding_windows_batch(
        raw_proc,
        window_seconds=epoch_seconds,
        step_seconds=sliding_step_seconds,
    )

    if X_sliding is not None and window_starts_abs is not None and len(X_sliding) > 0:
        X_sliding_proc, _ = epoch_pipeline.process_np(
            X_sliding,
            np.zeros(len(X_sliding)),
            shuffle=False,
        )
        sliding_labels, sliding_probs = model.predict_batch(X_sliding_proc)

        n_times_sliding = int(X_sliding.shape[2])
        window_starts_rel = np.clip(
            np.asarray(window_starts_abs, dtype=np.int64) - int(raw_proc.first_samp),
            0,
            len(sample_track) - 1,
        )
        window_ends_rel = np.clip(
            window_starts_rel + n_times_sliding - 1,
            0,
            len(sample_track) - 1,
        )

        for pred_label, prob_dict, end_rel_idx in zip(sliding_labels, sliding_probs, window_ends_rel):
            sliding_predictions.append(
                OfflinePrediction(
                    prediction=str(pred_label),
                    last_sample=int(sample_track[int(end_rel_idx)]),
                    probs=prob_dict,
                )
            )

    event_samples, event_desc = _extract_events_by_sample(raw_proc, sample_track)

    print(f"[Offline] Predicciones por eventos generadas: {len(predictions)}")
    if predictions:
        n_correct = sum(1 for p in predictions if p.is_correct)
        acc = 100.0 * n_correct / len(predictions)
        print(f"[Offline] Aciertos: {n_correct}/{len(predictions)} ({acc:.1f}%)")

    _plot_predictions_vs_events(
        predictions=predictions,
        sliding_predictions=sliding_predictions,
        event_samples=event_samples,
        event_desc=event_desc,
        sfreq=float(raw_proc.info["sfreq"]),
        class_order=lista,
    )

    return {
        "ea_matrix": ea_matrix,
        "model": model,
        "predictions": predictions,
        "sliding_predictions": sliding_predictions,
        "event_samples": event_samples,
        "event_desc": event_desc,
    }

def main():
    pretrainingFif = ["EEG_controller_app/recordings/suj2_1_raw.fif"]
    pretrainingFif += ["EEG_controller_app/recordings/suj2_2_raw.fif"]
    testFif = "EEG_controller_app/recordings/suj2_3_raw.fif"

    experimentoOffline(
        pretrainingFif=pretrainingFif,
        testFif=testFif,
    )

main()