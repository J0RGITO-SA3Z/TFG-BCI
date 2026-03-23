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


@dataclass
class OfflinePrediction:
    prediction: str
    last_sample: int
    probs: dict


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
    event_samples: np.ndarray,
    event_desc: list[str],
    sfreq: float,
    class_order: Sequence[str] | None = None,
    window_seconds: float = 30.0,
):
    if not predictions:
        print("No hay predicciones para mostrar.")
        return

    pred_samples = np.array([p.last_sample for p in predictions], dtype=np.int64)
    pred_labels = [p.prediction for p in predictions]

    if class_order:
        # Mantiene los valores originales de etiqueta en el eje Y.
        classes = list(dict.fromkeys(class_order))
        for label in pred_labels:
            if label not in classes:
                classes.append(label)
    else:
        classes = sorted(set(pred_labels))

    class_to_y = {c: i for i, c in enumerate(classes)}
    pred_y = np.array([class_to_y[c] for c in pred_labels], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))
    plt.subplots_adjust(bottom=0.25)

    scatter = ax.scatter(pred_samples, pred_y, s=18, alpha=0.85, label="Predicciones")

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

    legend_handles = [scatter]
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
    ax.set_title("Predicciones offline (ventana deslizante) vs eventos del testFif")
    ax.grid(alpha=0.25)

    full_min = int(min(pred_samples.min(), event_samples.min()) if event_samples.size else pred_samples.min())
    full_max = int(max(pred_samples.max(), event_samples.max()) if event_samples.size else pred_samples.max())

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


def experimentoOffline(
    pretrainingFif: str | Sequence[str],
    testFif: str,
    lista = ["left_hand", "right_hand"],
    epochs: int = 10,
    seed: int = 42,
    epoch_seconds: float = 4.0,
    step_seconds: float = 0.2,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
):
    """
    1) Entrena offline con pretrainingFif.
    2) Simula tiempo real sobre testFif con ventana deslizante.
    3) Devuelve predicciones alineadas por numero de muestra y muestra grafico con slider.
    """
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
    sample_track = _build_sample_track(raw_original)

    raw_pipeline = RawProcessorPipeline([
        BandpassFilter(l_freq=l_freq, h_freq=h_freq),
    ])
    raw_proc = raw_pipeline.process(raw_original)

    eeg_data = raw_proc.get_data()
    sfreq = float(raw_proc.info["sfreq"])

    epoch_samples = int(round(epoch_seconds * sfreq))
    step_samples = int(round(step_seconds * sfreq))

    if epoch_samples <= 0 or step_samples <= 0:
        raise ValueError("epoch_seconds y step_seconds deben generar al menos 1 muestra")

    if eeg_data.shape[1] < epoch_samples:
        raise ValueError("testFif no tiene suficientes muestras para una ventana completa")

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(matrix=ea_matrix),
        SpatialInterpolator(actual_channel_positions=[ch.upper() for ch in raw_proc.ch_names]),
    ])

    print("[Offline] Simulando predicciones cada", step_seconds, "segundos...")
    predictions: list[OfflinePrediction] = []

    for end in range(epoch_samples, eeg_data.shape[1] + 1, step_samples):
        window = eeg_data[:, end - epoch_samples : end]
        X_proc, _ = epoch_pipeline.process_np(np.asarray([window]), np.asarray([0]), shuffle=False)
        pred_label, prob_dict = model.predict(X_proc[0])

        last_sample = int(sample_track[end - 1])
        predictions.append(
            OfflinePrediction(
                prediction=pred_label,
                last_sample=last_sample,
                probs=prob_dict,
            )
        )

    event_samples, event_desc = _extract_events_by_sample(raw_original, sample_track)

    print(f"[Offline] Predicciones generadas: {len(predictions)}")
    _plot_predictions_vs_events(
        predictions=predictions,
        event_samples=event_samples,
        event_desc=event_desc,
        sfreq=sfreq,
        class_order=lista,
    )

    return {
        "ea_matrix": ea_matrix,
        "model": model,
        "predictions": predictions,
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