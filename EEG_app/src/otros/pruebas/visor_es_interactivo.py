"""
Visor interactivo para explorar parámetros de Exponential Smoothing sin necesidad
de re-ejecutar la simulación de tiempo real.

Lee el CSV generado por RT_interpreter_slider (requiere columnas:
  sample, p_left_raw, p_right_raw, alpha, threshold)
y permite ajustar alpha y threshold en tiempo real mediante sliders,
recomputando el suavizado y las predicciones al instante.

Si el FIF existe calcula y muestra la accuracy en el título.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
from matplotlib.widgets import Slider

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── RUTAS ─────────────────────────────────────────────────────────────────
SIM_DIR  = os.path.join(PROJECT_ROOT, "recordings", "simulations_RT")

CSV_PATH = os.path.join(SIM_DIR, "[NEW]ES_2_6_85_75", "SalidaPredicciones.csv")
FIF_PATH = os.path.join(SIM_DIR, "[NEW]ES_2_6_85_75", "suj2_6_raw.fif")

# ── CONSTANTES ────────────────────────────────────────────────────────────
ANNOTATION_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}
TRIAL_COLORS = {
    "left_hand":  "tab:blue",
    "right_hand": "tab:orange",
    "feet":       "tab:green",
    "rest":       "tab:gray",
}
PRED_COLORS = {
    "left_hand":  "tab:blue",
    "right_hand": "tab:orange",
    "rest":       "tab:gray",
    None:         "lightgray",
}

TRIAL_DURATION_S  = 4.0
ACCURACY_OFFSET_S = 4.0     # busca la predicción en trial_start + este offset
CLASES_ACCURACY   = ["left_hand", "right_hand"]
DOT_COLOR         = "red"
DOT_SIZE          = 18


# ── ES: recomputar ─────────────────────────────────────────────────────────

def _recompute_smooth(p_raw: np.ndarray, alpha: float) -> np.ndarray:
    """Replica exactamente la fórmula de ExponentialSmoothing.decider():
       smooth[i] = alpha * smooth[i-1] + (1-alpha) * raw[i]
       (alpha = peso del pasado)."""
    n = len(p_raw)
    smooth = np.empty(n)
    smooth[0] = p_raw[0]
    for i in range(1, n):
        smooth[i] = alpha * smooth[i - 1] + (1.0 - alpha) * p_raw[i]
    return smooth


def _recompute_predictions(
    p_left_smooth: np.ndarray,
    p_right_smooth: np.ndarray,
    threshold: float,
) -> list[str]:
    """Replica la lógica de decisión de ExponentialSmoothing.decider():
       right_hand tiene prioridad sobre left_hand."""
    preds = []
    for l, r in zip(p_left_smooth, p_right_smooth):
        if r > threshold:
            preds.append("right_hand")
        elif l > threshold:
            preds.append("left_hand")
        else:
            preds.append("rest")
    return preds


# ── FIF helpers ────────────────────────────────────────────────────────────

def _build_sample_track(raw: mne.io.BaseRaw) -> np.ndarray:
    if "Sample" in raw.ch_names:
        return np.asarray(raw.get_data(picks=["Sample"])[0], dtype=np.int64)
    return np.arange(raw.first_samp, raw.first_samp + raw.n_times, dtype=np.int64)


def _load_fif(fif_path: str) -> tuple[np.ndarray, list[str], float]:
    if fif_path and os.path.exists(fif_path):
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        track = _build_sample_track(raw)
        if len(raw.annotations) == 0:
            return np.array([], dtype=np.int64), [], sfreq
        idx = raw.time_as_index(raw.annotations.onset, use_rounding=True)
        idx = np.clip(np.asarray(idx, dtype=np.int64), 0, len(track) - 1)
        ev_samples = track[idx]
        ev_desc = [ANNOTATION_MAP.get(d, d) for d in raw.annotations.description]
        print(f"FIF cargado — {len(ev_samples)} eventos.")
        return ev_samples, ev_desc, sfreq
    print(f"⚠️  FIF no encontrado (se omite accuracy): {fif_path}")
    return np.array([], dtype=np.int64), [], 250.0


# ── Accuracy ───────────────────────────────────────────────────────────────

def _compute_accuracy(
    samples: np.ndarray,
    predictions: list[str],
    ev_samples: np.ndarray,
    ev_desc: list[str],
    sfreq: float,
) -> dict | None:
    """Para cada trial busca la predicción más cercana a trial_start + ACCURACY_OFFSET_S."""
    if ev_samples.size == 0:
        return None

    offset = int(ACCURACY_OFFSET_S * sfreq)
    correct, total = 0, 0
    per_class: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for s, desc in zip(ev_samples, ev_desc):
        if desc not in CLASES_ACCURACY:
            continue
        target = int(s) + offset
        closest = int(np.argmin(np.abs(samples - target)))
        pred = predictions[closest]
        per_class_total[desc] = per_class_total.get(desc, 0) + 1
        total += 1
        if pred == desc:
            correct += 1
            per_class[desc] = per_class.get(desc, 0) + 1

    if total == 0:
        return None
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "per_class_total": per_class_total,
    }


def _format_accuracy(acc: dict | None, prefix: str = "") -> str:
    if acc is None:
        return ""
    per = "  ".join(
        f"{cls}: {acc['per_class'].get(cls, 0)}/{acc['per_class_total'].get(cls, 0)}"
        for cls in CLASES_ACCURACY
    )
    return f"{prefix}Acc {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})  [{per}]"


# ── Helpers de dibujo ──────────────────────────────────────────────────────

def _shade_trials(ax, ev_samples, ev_desc, trial_samples: int) -> None:
    for s, desc in zip(ev_samples, ev_desc):
        color = TRIAL_COLORS.get(desc, "gray")
        ax.axvspan(s, s + trial_samples, color=color, alpha=0.07, linewidth=0)


def _shade_predictions_batched(ax, samples: np.ndarray, predictions: list[str]) -> None:
    """Agrupa tramos consecutivos con la misma predicción en un único axvspan."""
    if len(samples) < 2:
        return
    step = float(samples[1] - samples[0])
    half = step / 2.0
    i = 0
    while i < len(predictions):
        pred = predictions[i]
        color = PRED_COLORS.get(pred, "lightgray")
        j = i + 1
        while j < len(predictions) and predictions[j] == pred:
            j += 1
        ax.axvspan(samples[i] - half, samples[j - 1] + half,
                   color=color, alpha=0.15, linewidth=0)
        i = j


_EVENT_HIDDEN = {"feet", "cross"}


def _draw_event_lines(ax, ev_samples, ev_desc) -> None:
    """Oculta feet, cross y el blank inmediatamente posterior a cada feet."""
    desc_list = list(ev_desc)
    feet_blank_idx = {
        i + 1
        for i, d in enumerate(desc_list[:-1])
        if d.lower() == "feet" and desc_list[i + 1].lower() == "blank"
    }
    pairs = [
        (s, d) for i, (s, d) in enumerate(zip(ev_samples, desc_list))
        if d.lower() not in _EVENT_HIDDEN and i not in feet_blank_idx
    ]
    if not pairs:
        return
    unique = list(dict.fromkeys(d for _, d in pairs))
    cmap   = plt.get_cmap("tab10")
    cmap_d = {d: cmap(k % cmap.N) for k, d in enumerate(unique)}
    trans  = ax.get_xaxis_transform()
    for s, desc in pairs:
        c = cmap_d[desc]
        ax.axvline(s, color=c, alpha=0.6, linewidth=1.2)
        ax.text(s, 0.97, desc, rotation=90, va="top", ha="right",
                fontsize=7, color=c, alpha=0.85, transform=trans)


def _scatter_pred_dots(ax, samples: np.ndarray, values: np.ndarray, predictions: list) -> None:
    """Scatter dots coloreados por clase de predicción."""
    pred_arr = np.array([str(p) for p in predictions])
    for pred in dict.fromkeys(str(p) for p in predictions):
        mask = pred_arr == pred
        color = PRED_COLORS.get(pred, "lightgray")
        ax.scatter(samples[mask], values[mask], s=DOT_SIZE, color=color, zorder=5, linewidths=0)


def _draw_panel(
    ax, samples, raw, smooth, predictions,
    ev_samples, ev_desc, trial_samples, threshold, color, ylabel,
) -> None:
    ax.cla()
    ax.set_facecolor("white")
    # Probabilidades crudas: solo puntos scatter, sin línea
    ax.scatter(samples, raw, s=DOT_SIZE * 0.4, color="red", alpha=0.85,
               zorder=2, label="raw", linewidths=0)
    # Señal suavizada: línea + puntos coloreados por predicción
    ax.plot(samples, smooth, color=color, linewidth=1.4, alpha=0.9, label="smooth")
    _scatter_pred_dots(ax, samples, smooth, predictions)
    ax.axhline(threshold, color=color, linestyle="--", linewidth=2.2, alpha=1.0,
               label=f"threshold ({threshold:.2f})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    _draw_event_lines(ax, ev_samples, ev_desc)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


# ── Función principal ──────────────────────────────────────────────────────

def visor_es_interactivo(csv_path: str = CSV_PATH, fif_path: str = FIF_PATH) -> None:
    # ── Cargar CSV ─────────────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        print(f"❌ CSV no encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    required = {"sample", "p_left_raw", "p_right_raw"}
    missing  = required - set(df.columns)
    if missing:
        print(f"❌ El CSV no tiene las columnas necesarias: {missing}")
        print(f"   Columnas presentes: {list(df.columns)}")
        print("   El CSV debe ser generado con RT_interpreter_slider (formato nuevo).")
        return

    samples     = df["sample"].to_numpy(dtype=np.int64)
    p_left_raw  = df["p_left_raw"].to_numpy(dtype=float)
    p_right_raw = df["p_right_raw"].to_numpy(dtype=float)

    orig_alpha     = float(df["alpha"].iloc[0])     if "alpha"     in df.columns else 0.85
    orig_threshold = float(df["threshold"].iloc[0]) if "threshold" in df.columns else 0.75
    print(f"CSV cargado — {len(samples)} predicciones  |  α={orig_alpha}  threshold={orig_threshold}")

    # ── Cargar FIF ─────────────────────────────────────────────────────────
    ev_samples, ev_desc, sfreq = _load_fif(fif_path)
    trial_samples = int(TRIAL_DURATION_S * sfreq)

    # Accuracy con parámetros originales del CSV (baseline)
    orig_preds      = df["prediction"].tolist() if "prediction" in df.columns else None
    orig_acc_str    = ""
    if orig_preds is not None and ev_samples.size > 0:
        orig_acc = _compute_accuracy(samples, orig_preds, ev_samples, ev_desc, sfreq)
        orig_acc_str = _format_accuracy(orig_acc, prefix="CSV orig → ")

    # ── Figura ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(17, 9), sharex=True)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(bottom=0.18, hspace=0.08)

    ax_sl_alpha = fig.add_axes([0.12, 0.08, 0.33, 0.03])
    ax_sl_thr   = fig.add_axes([0.55, 0.08, 0.33, 0.03])

    slider_alpha = Slider(
        ax_sl_alpha, "α  (peso pasado)",
        valmin=0.01, valmax=0.99, valinit=orig_alpha, valstep=0.01,
        color="steelblue",
    )
    slider_thr = Slider(
        ax_sl_thr, "threshold",
        valmin=0.50, valmax=0.99, valinit=orig_threshold, valstep=0.01,
        color="darkorange",
    )

    # ── Callback ───────────────────────────────────────────────────────────
    _initialized = [False]

    def _update(_val):
        # Preservar zoom/pan: guardar xlim antes de limpiar los ejes
        xlim = axes[0].get_xlim() if _initialized[0] else None

        alpha     = slider_alpha.val
        threshold = slider_thr.val

        p_left_smooth  = _recompute_smooth(p_left_raw,  alpha)
        p_right_smooth = _recompute_smooth(p_right_raw, alpha)
        preds          = _recompute_predictions(p_left_smooth, p_right_smooth, threshold)

        _draw_panel(axes[0], samples, p_left_raw,  p_left_smooth,  preds,
                    ev_samples, ev_desc, trial_samples, threshold, "tab:blue",   "p left_hand")
        _draw_panel(axes[1], samples, p_right_raw, p_right_smooth, preds,
                    ev_samples, ev_desc, trial_samples, threshold, "tab:orange", "p right_hand")
        axes[1].set_xlabel("Número de muestra")

        # Restaurar zoom/pan tras el redibujado
        if xlim is not None:
            axes[0].set_xlim(xlim)

        _initialized[0] = True

        # Título con accuracy nueva y baseline
        acc     = _compute_accuracy(samples, preds, ev_samples, ev_desc, sfreq)
        acc_str = _format_accuracy(acc, prefix="Nuevo → ")
        title   = f"ES Interactivo  |  α={alpha:.2f}  |  threshold={threshold:.2f}"
        if acc_str:
            title += f"\n{acc_str}"
        if orig_acc_str:
            title += f"   ||   {orig_acc_str}"
        fig.suptitle(title, fontsize=10)

        # Leyenda global: puntos coloreados por predicción
        pred_patches = [
            mpatches.Patch(color=PRED_COLORS["left_hand"],  label="pred: left_hand"),
            mpatches.Patch(color=PRED_COLORS["right_hand"], label="pred: right_hand"),
            mpatches.Patch(color=PRED_COLORS["rest"],       label="pred: rest"),
        ]
        fig.legend(
            handles=pred_patches,
            loc="lower center", ncol=3,
            fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.0),
        )
        fig.canvas.draw_idle()

    slider_alpha.on_changed(_update)
    slider_thr.on_changed(_update)

    _update(None)   # dibujo inicial
    plt.show()


if __name__ == "__main__":
    visor_es_interactivo()
