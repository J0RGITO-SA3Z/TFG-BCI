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
    "[NEW]Media_Probs_2_6", "SalidaPredicciones.csv"
)
FIF_PATH = os.path.join(
    PROJECT_ROOT, "recordings", "simulations_RT",
    "[NEW]Media_Probs_2_6", "suj2_6_raw.fif"
)

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

TRIAL_DURATION_S = 4.0


# ── helpers FIF ───────────────────────────────────────────────────────────

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
    return sample_track[event_idx], list(raw.annotations.description)


# ── visor principal ───────────────────────────────────────────────────────

def visor_media_probabilidades(
    csv_path: str = CSV_PATH,
    fif_path: str = FIF_PATH,
) -> None:
    if not os.path.exists(csv_path):
        print(f"❌ No se encuentra el CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required = {"sample", "prediction", "p_left_raw", "p_right_raw", "mean_left", "mean_right"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ El CSV no tiene las columnas necesarias: {missing}")
        print(f"   Columnas encontradas: {list(df.columns)}")
        return

    samples     = df["sample"].to_numpy()
    p_left_raw  = df["p_left_raw"].to_numpy()
    p_right_raw = df["p_right_raw"].to_numpy()
    mean_left   = df["mean_left"].to_numpy()
    mean_right  = df["mean_right"].to_numpy()
    predictions = df["prediction"].tolist()

    # ── FIF ───────────────────────────────────────────────────────────────
    event_samples = np.array([], dtype=np.int64)
    event_desc: list[str] = []
    sfreq = 250.0

    if fif_path and os.path.exists(fif_path):
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        sample_track = _build_sample_track(raw)
        event_samples, event_desc = _extract_events_by_sample(raw, sample_track)
        event_desc = [ANNOTATION_MAP.get(d, d) for d in event_desc]
        print(f"FIF cargado — {len(event_samples)} eventos.")
    else:
        print("⚠️  FIF no encontrado — se omiten los eventos.")

    trial_samples = int(TRIAL_DURATION_S * sfreq)

    # ── figura: 2 paneles ─────────────────────────────────────────────────
    # Igual que exponential smoothing pero mostrando media de ventana vs raw
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.suptitle("Media de Probabilidades de Ventana  |  raw vs media deslizante", fontsize=12)

    for ax, raw_probs, mean_probs, ylabel, color in [
        (axes[0], p_left_raw,  mean_left,  "Probabilidad  left_hand",  "tab:blue"),
        (axes[1], p_right_raw, mean_right, "Probabilidad  right_hand", "tab:orange"),
    ]:
        _shade_trials(ax, event_samples, event_desc, trial_samples)
        _shade_predictions(ax, samples, predictions)

        ax.plot(samples, raw_probs,  color=color, alpha=0.35, linewidth=1.0,
                linestyle="--", label="raw (última muestra de ventana)")
        ax.plot(samples, mean_probs, color=color, alpha=0.95, linewidth=1.8,
                label="media de ventana")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
                   label="0.5 (umbral natural)")

        _draw_event_lines(ax, event_samples, event_desc)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[1].set_xlabel("Número de muestra")

    # ── leyenda global ────────────────────────────────────────────────────
    pred_patches = [
        mpatches.Patch(color=PRED_COLORS["left_hand"],  alpha=0.3, label="pred: left_hand"),
        mpatches.Patch(color=PRED_COLORS["right_hand"], alpha=0.3, label="pred: right_hand"),
        mpatches.Patch(color=PRED_COLORS["rest"],       alpha=0.3, label="pred: rest"),
    ]
    trial_patches = [
        mpatches.Patch(color=TRIAL_COLORS.get(cls, "gray"), alpha=0.15, label=f"trial: {cls}")
        for cls in dict.fromkeys(event_desc)
        if cls in TRIAL_COLORS
    ]
    fig.legend(handles=pred_patches + trial_patches, loc="lower center",
               ncol=len(pred_patches) + len(trial_patches),
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

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


def _shade_trials(ax, event_samples, event_desc, trial_samples) -> None:
    for sample, desc in zip(event_samples, event_desc):
        color = TRIAL_COLORS.get(desc, "gray")
        ax.axvspan(sample, sample + trial_samples, color=color, alpha=0.07, linewidth=0)


def _draw_event_lines(ax, event_samples, event_desc) -> None:
    unique_descs = list(dict.fromkeys(event_desc))
    cmap = plt.get_cmap("tab10")
    color_map = {d: cmap(i % cmap.N) for i, d in enumerate(unique_descs)}
    for sample, desc in zip(event_samples, event_desc):
        color = color_map[desc]
        ax.axvline(sample, color=color, alpha=0.6, linewidth=1.2)
        ax.text(sample, ax.get_ylim()[1] * 0.98, desc, rotation=90,
                va="top", ha="right", fontsize=7, color=color, alpha=0.85)


if __name__ == "__main__":
    visor_media_probabilidades()
