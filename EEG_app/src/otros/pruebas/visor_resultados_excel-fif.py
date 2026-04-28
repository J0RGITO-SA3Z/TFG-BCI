import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from dataclasses import dataclass
from typing import Sequence
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

# --- Configuración de Rutas y Proyecto ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR,"..", "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Rutas a tus archivos de datos guardados
EXCEL_PATH = os.path.join(PROJECT_ROOT, "recordings", "simulations_RT","SW_Exponential_Smoothing", "SalidaPredicciones.csv")
FIF_PATHS = [os.path.join(PROJECT_ROOT, "recordings", "simulations_RT", "SW_Exponential_Smoothing","suj2_2_raw.fif")]

CLASES = ["left_hand", "right_hand", "feet", "rest"]

ANNOTATION_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

# =====================================================================
# 1. CÓDIGO EXTRAÍDO LITERALMENTE DE TU 'experimento_Offline.py'
# =====================================================================

@dataclass
class OfflinePrediction:
    prediction: str
    last_sample: int
    probs: dict
    true_label: str | None = None
    is_correct: bool | None = None

# Devuelve el numero de muestra para la posicion en el array de muestras del raw
def _build_sample_track(raw: mne.io.BaseRaw) -> np.ndarray:
    if "Sample" in raw.ch_names:
        sample_track = raw.get_data(picks=["Sample"])[0]
        return np.asarray(sample_track, dtype=np.int64)
    return np.arange(raw.first_samp, raw.first_samp + raw.n_times, dtype=np.int64)

# Saca una lista de los posicion de muestra en la que hay eventos y la lista de nombres de dichos eventos
def _extract_events_by_sample(raw: mne.io.BaseRaw, sample_track: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if len(raw.annotations) == 0:
        return np.array([], dtype=np.int64), []

    event_idx = raw.time_as_index(raw.annotations.onset, use_rounding=True)
    event_idx = np.asarray(event_idx, dtype=np.int64)
    event_idx = np.clip(event_idx, 0, len(sample_track) - 1)

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
    accuracy_info: dict | None = None,
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
    if accuracy_info:
        acc_str = f"  —  Accuracy: {accuracy_info['correct']}/{accuracy_info['total']} ({accuracy_info['accuracy']:.1%})"
        per_class_lines = "   ".join(
            f"{cls}: {accuracy_info['per_class_correct'].get(cls, 0)}/{accuracy_info['per_class_total'].get(cls, 0)}"
            f" ({accuracy_info['per_class_correct'].get(cls, 0) / accuracy_info['per_class_total'][cls]:.1%})"
            for cls in accuracy_info['per_class_total']
        )
        ax.set_title(
            f"Visor Offline de RT: Predicciones vs Eventos Reales{acc_str}\n{per_class_lines}",
            fontsize=9,
        )
    else:
        ax.set_title("Visor Offline de RT: Predicciones vs Eventos Reales")
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
        label="Inicio ventana",
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

# =====================================================================
# 2. CÁLCULO DE ACCURACY (predicción a los 4 s de cada marca)
# =====================================================================

def _calcular_accuracy(
    sliding_predictions: list[OfflinePrediction],
    event_samples: np.ndarray,
    event_desc: list[str],
    clases: list[str],
    sfreq: float,
    offset_segundos: float = 4.0,
) -> dict:
    """
    Para cada evento cuya descripción esté en *clases*, busca la predicción
    más cercana a S_i + offset_segundos*sfreq y la compara con la etiqueta real.
    """
    from collections import Counter

    # Solo trials que correspondan a clases conocidas
    trials = [
        (int(s), desc)
        for s, desc in zip(event_samples, event_desc)
        if desc in clases
    ]

    if not trials:
        print("⚠️  No se encontraron eventos de clases conocidas para calcular accuracy.")
        return {}

    pred_samples = np.array([p.last_sample for p in sliding_predictions], dtype=np.int64)
    pred_labels  = [p.prediction for p in sliding_predictions]

    correct = 0
    total   = 0
    per_class_correct = Counter()
    per_class_total   = Counter()

    offset_samples = int(offset_segundos * sfreq)

    for sample_start, true_label in trials:
        target_sample = sample_start + offset_samples

        if len(pred_samples) == 0:
            continue

        # Predicción más cercana al instante objetivo
        closest_idx = int(np.argmin(np.abs(pred_samples - target_sample)))
        predicted_label = pred_labels[closest_idx]
        is_correct = predicted_label == true_label

        total += 1
        per_class_total[true_label] += 1
        if is_correct:
            correct += 1
            per_class_correct[true_label] += 1

    if total == 0:
        print("⚠️  Ningún trial tenía predicciones en su ventana.")
        return {}

    accuracy = correct / total
    print(f"\n{'─'*45}")
    print(f"  Accuracy global: {correct}/{total}  →  {accuracy:.1%}")
    for cls in clases:
        if per_class_total[cls]:
            print(f"    {cls:>12}: {per_class_correct[cls]}/{per_class_total[cls]}"
                  f"  ({per_class_correct[cls]/per_class_total[cls]:.1%})")
    print(f"{'─'*45}\n")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_correct": dict(per_class_correct),
        "per_class_total": dict(per_class_total),
    }


def _calcular_accuracy_voto(
    sliding_predictions: list[OfflinePrediction],
    event_samples: np.ndarray,
    event_desc: list[str],
    clases: list[str],
    sfreq: float,
    offset_segundos: float = 4.0,
    ventana_segundos: float = 2.0,
) -> dict:
    """
    Para cada evento cuya descripción esté en *clases*, recoge todas las
    predicciones en el intervalo [S_i + offset*sfreq, S_i + (offset+ventana)*sfreq]
    y toma la clase más frecuente (voto por mayoría) como predicción final.
    Por defecto: ventana de 2 s justo después de los 4 s del epoch ([+4s, +6s]).
    """
    from collections import Counter

    trials = [
        (int(s), desc)
        for s, desc in zip(event_samples, event_desc)
        if desc in clases
    ]

    if not trials:
        print("⚠️  [Voto] No se encontraron eventos de clases conocidas.")
        return {}

    pred_samples = np.array([p.last_sample for p in sliding_predictions], dtype=np.int64)
    pred_labels  = [p.prediction for p in sliding_predictions]

    offset_samples  = int(offset_segundos * sfreq)
    ventana_samples = int(ventana_segundos * sfreq)

    correct = 0
    total   = 0
    per_class_correct = Counter()
    per_class_total   = Counter()
    skipped = 0

    for sample_start, true_label in trials:
        win_ini = sample_start + offset_samples
        win_fin = win_ini + ventana_samples

        mask = (pred_samples >= win_ini) & (pred_samples <= win_fin)
        labels_en_ventana = [pred_labels[i] for i in np.where(mask)[0]]

        if not labels_en_ventana:
            skipped += 1
            continue

        predicted_label = Counter(labels_en_ventana).most_common(1)[0][0]
        is_correct = predicted_label == true_label

        total += 1
        per_class_total[true_label] += 1
        if is_correct:
            correct += 1
            per_class_correct[true_label] += 1

    if total == 0:
        print("⚠️  [Voto] Ningún trial tenía predicciones en su ventana.")
        return {}

    accuracy = correct / total
    print(f"\n{'─'*45}")
    print(f"  [Voto +{offset_segundos:.1f}s..+{offset_segundos+ventana_segundos:.1f}s] "
          f"Accuracy global: {correct}/{total}  →  {accuracy:.1%}"
          + (f"  ({skipped} trials sin predicciones)" if skipped else ""))
    for cls in clases:
        if per_class_total[cls]:
            print(f"    {cls:>12}: {per_class_correct[cls]}/{per_class_total[cls]}"
                  f"  ({per_class_correct[cls]/per_class_total[cls]:.1%})")
    print(f"{'─'*45}\n")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_correct": dict(per_class_correct),
        "per_class_total": dict(per_class_total),
    }




# =====================================================================
# 3. NUEVA LÓGICA PARA LEER EXCEL Y FIF, Y ENVIARLO A TU GRÁFICA
# =====================================================================

def analizar_rt_offline():
    print(f"--- Cargando FIF: {FIF_PATHS[0]} ---")
    raw = mne.io.read_raw_fif(FIF_PATHS[0], preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    
    # Extraemos eventos del FIF exactamente como en tu script
    sample_track = _build_sample_track(raw)
    event_samples, event_desc = _extract_events_by_sample(raw, sample_track)
    event_desc = [ANNOTATION_MAP.get(d, d) for d in event_desc]

    print(f"--- Cargando CSV: {EXCEL_PATH} ---")
    if not os.path.exists(EXCEL_PATH):
        print("❌ No se encuentra el archivo CSV.")
        return

    df = pd.read_csv(EXCEL_PATH)

    sliding_predictions = []

    # Convertimos cada fila del CSV en tu objeto OfflinePrediction
    for _, row in df.iterrows():
        pred_label = str(row['prediction'])

        # Si el CSV no tuviera la columna sample por lo que sea, la estimamos
        if 'sample' in row:
            sample_val = int(row['sample'])
        else:
            sample_val = 0

        # Metemos las predicciones del CSV como "sliding predictions"
        # (que son los puntitos azules de tu gráfica)
        sliding_predictions.append(
            OfflinePrediction(
                prediction=pred_label,
                last_sample=sample_val,
                probs={}  # El plot no las necesita visualmente
            )
        )

    print(f"✅ Cargadas {len(sliding_predictions)} predicciones de CSV y {len(event_samples)} eventos del FIF.")

    # ------------------------------------------------------------------
    # Método 1: predicción más cercana a marca + 4 s
    # ------------------------------------------------------------------
    accuracy_info = _calcular_accuracy(sliding_predictions, event_samples, event_desc, CLASES, sfreq)

    # ------------------------------------------------------------------
    # Método 2: voto por mayoría en ventana [marca+4s, marca+6s]
    # ------------------------------------------------------------------
    _calcular_accuracy_voto(sliding_predictions, event_samples, event_desc, CLASES, sfreq)


    print("--- Abriendo visor interactivo... ---")

    # Llamamos a tu función de graficado
    _plot_predictions_vs_events(
        predictions=[],
        sliding_predictions=sliding_predictions,
        event_samples=event_samples,
        event_desc=event_desc,
        sfreq=sfreq,
        class_order=CLASES,
        accuracy_info=accuracy_info,
    )

if __name__ == "__main__":
    analizar_rt_offline()