import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mne

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(
    PROJECT_ROOT, "recordings", "simulations_RT",
    "[NEW]ES_2_6_85_75", "SalidaPredicciones.csv"
)
FIF_PATH = os.path.join(
    PROJECT_ROOT, "recordings", "simulations_RT",
    "[NEW]ES_2_6_85_75", "suj2_6_raw.fif"
)

ANNOTATION_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

# color del fondo de cada trial según clase real pedida
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

TRIAL_DURATION_S = 4.0  # segundos que dura cada trial tras la marca


# ── helpers FIF (igual que en visor_resultados_excel-fif) ─────────────────

def _build_sample_track(raw: mne.io.BaseRaw) -> np.ndarray:
    if "Sample" in raw.ch_names:
        return np.asarray(raw.get_data(picks=["Sample"])[0], dtype=np.int64)
    return np.arange(raw.first_samp, raw.first_samp + raw.n_times, dtype=np.int64)


def _extract_events_by_sample(
    raw: mne.io.BaseRaw, sample_track: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    if len(raw.annotations) == 0:
        return np.array([], dtype=np.int64), []

    event_idx = raw.time_as_index(raw.annotations.onset, use_rounding=True)
    event_idx = np.clip(np.asarray(event_idx, dtype=np.int64), 0, len(sample_track) - 1)
    event_samples = sample_track[event_idx]
    event_desc = list(raw.annotations.description)
    return event_samples, event_desc


# ── visor principal ───────────────────────────────────────────────────────

def visor_exponential_smoothing(
    csv_path: str = CSV_PATH,
    fif_path: str = FIF_PATH,
) -> None:
    # ── CSV ───────────────────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        print(f"❌ No se encuentra el CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required = {"sample", "prediction", "p_left_raw", "p_right_raw",
                "p_left_smooth", "p_right_smooth"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ El CSV no tiene las columnas necesarias: {missing}")
        print(f"   Columnas encontradas: {list(df.columns)}")
        return

    threshold = float(df["threshold"].iloc[0]) if "threshold" in df.columns else None
    alpha     = float(df["alpha"].iloc[0])     if "alpha"     in df.columns else None

    samples        = df["sample"].to_numpy()
    p_left_raw     = df["p_left_raw"].to_numpy()
    p_right_raw    = df["p_right_raw"].to_numpy()
    p_left_smooth  = df["p_left_smooth"].to_numpy()
    p_right_smooth = df["p_right_smooth"].to_numpy()
    predictions    = df["prediction"].tolist()

    # ── FIF (opcional) ────────────────────────────────────────────────────
    event_samples = np.array([], dtype=np.int64)
    event_desc: list[str] = []
    sfreq = 250.0  # fallback si no hay FIF

    if fif_path and os.path.exists(fif_path):
        print(f"Cargando FIF: {fif_path}")
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        sample_track = _build_sample_track(raw)
        event_samples, event_desc = _extract_events_by_sample(raw, sample_track)
        event_desc = [ANNOTATION_MAP.get(d, d) for d in event_desc]
        print(f"  {len(event_samples)} eventos encontrados.")
    else:
        print("⚠️  FIF no encontrado — se omiten los eventos.")

    # ── figura ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(
        f"Exponential Smoothing  |  α={alpha}  |  threshold={threshold}",
        fontsize=12,
    )

    trial_samples = int(TRIAL_DURATION_S * sfreq)

    for ax, raw_probs, smooth_probs, ylabel, color in [
        (axes[0], p_left_raw,  p_left_smooth,  "Probabilidad  left_hand",  "tab:blue"),
        (axes[1], p_right_raw, p_right_smooth, "Probabilidad  right_hand", "tab:orange"),
    ]:
        # zonas de trial (fondo muy suave)
        _shade_trials(ax, event_samples, event_desc, trial_samples)

        # zonas de predicción del filtro (fondo algo más marcado)
        _shade_predictions(ax, samples, predictions)

        # líneas de probabilidad
        ax.plot(samples, raw_probs,    color=color, alpha=0.35, linewidth=1.0,
                linestyle="--", label="raw")
        ax.plot(samples, smooth_probs, color=color, alpha=0.95, linewidth=1.8,
                label="smooth")

        # threshold
        if threshold is not None:
            ax.axhline(threshold, color=color, linestyle=":", linewidth=1.2, alpha=0.6,
                       label=f"threshold ({threshold})")

        # líneas verticales de eventos
        _draw_event_lines(ax, event_samples, event_desc)

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[1].set_xlabel("Número de muestra")

    # ── leyenda global ────────────────────────────────────────────────────
    pred_patches = [
        mpatches.Patch(color=PRED_COLORS["left_hand"],  alpha=0.3,  label="pred: left_hand"),
        mpatches.Patch(color=PRED_COLORS["right_hand"], alpha=0.3,  label="pred: right_hand"),
        mpatches.Patch(color=PRED_COLORS["rest"],       alpha=0.3,  label="pred: rest"),
    ]
    trial_patches = [
        mpatches.Patch(color=TRIAL_COLORS.get(cls, "gray"), alpha=0.15, label=f"trial: {cls}")
        for cls in dict.fromkeys(event_desc)  # orden de aparición, sin duplicados
        if cls in TRIAL_COLORS
    ]
    fig.legend(
        handles=pred_patches + trial_patches,
        loc="lower center", ncol=len(pred_patches) + len(trial_patches),
        fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


# ── helpers de dibujo ─────────────────────────────────────────────────────

def _shade_predictions(ax, samples: np.ndarray, predictions: list) -> None:
    if len(samples) < 2:
        return
    half_step = (samples[1] - samples[0]) / 2.0
    for i, pred in enumerate(predictions):
        color = PRED_COLORS.get(str(pred), "lightgray")
        ax.axvspan(samples[i] - half_step, samples[i] + half_step,
                   color=color, alpha=0.15, linewidth=0)


def _shade_trials(
    ax, event_samples: np.ndarray, event_desc: list[str], trial_samples: int
) -> None:
    """Fondo muy tenue durante la duración de cada trial."""
    for sample, desc in zip(event_samples, event_desc):
        color = TRIAL_COLORS.get(desc, "gray")
        ax.axvspan(sample, sample + trial_samples, color=color, alpha=0.07, linewidth=0)


def _draw_event_lines(
    ax, event_samples: np.ndarray, event_desc: list[str]
) -> None:
    """Línea vertical + etiqueta en cada marca del FIF."""
    unique_descs = list(dict.fromkeys(event_desc))
    cmap = plt.get_cmap("tab10")
    color_map = {d: cmap(i % cmap.N) for i, d in enumerate(unique_descs)}

    for sample, desc in zip(event_samples, event_desc):
        color = color_map[desc]
        ax.axvline(sample, color=color, alpha=0.6, linewidth=1.2)
        ax.text(sample, 1.02, desc, rotation=90, va="bottom", ha="right",
                fontsize=7, color=color, alpha=0.85)


if __name__ == "__main__":
    visor_exponential_smoothing()
