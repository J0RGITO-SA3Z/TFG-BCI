import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mne

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── RUTAS ─────────────────────────────────────────────────────────────────
# Modifica estas rutas para apuntar a las carpetas de cada simulación.

SIM_DIR = os.path.join(PROJECT_ROOT, "recordings", "simulations_RT")

CSV_FUGA  = os.path.join(SIM_DIR, "[NEW]IntegratorFuga_2_6",  "SalidaPredicciones.csv")
FIF_FUGA  = os.path.join(SIM_DIR, "[NEW]IntegratorFuga_2_6",  "suj2_6_raw.fif")

CSV_SCORE = os.path.join(SIM_DIR, "[NEW]ScorePonderado_2_6",  "SalidaPredicciones.csv")
FIF_SCORE = os.path.join(SIM_DIR, "[NEW]ScorePonderado_2_6",  "suj2_6_raw.fif")

CSV_MEDIA = os.path.join(SIM_DIR, "[NEW]Media_Probs_2_6",     "SalidaPredicciones.csv")
FIF_MEDIA = os.path.join(SIM_DIR, "[NEW]Media_Probs_2_6",     "suj2_6_raw.fif")

CSV_ES    = os.path.join(SIM_DIR, "[NEW]ES_2_6_85_75",        "SalidaPredicciones.csv")
FIF_ES    = os.path.join(SIM_DIR, "[NEW]ES_2_6_85_75",        "suj2_6_raw.fif")

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

TRIAL_DURATION_S = 4.0
DOT_COLOR        = "red"
DOT_SIZE         = 18


# ── HELPERS FIF ───────────────────────────────────────────────────────────

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


def _load_fif(fif_path: str) -> tuple[np.ndarray, list[str], float]:
    """Carga FIF y extrae eventos. Devuelve (event_samples, event_desc, sfreq)."""
    if fif_path and os.path.exists(fif_path):
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        track = _build_sample_track(raw)
        ev_samples, ev_desc = _extract_events_by_sample(raw, track)
        ev_desc = [ANNOTATION_MAP.get(d, d) for d in ev_desc]
        print(f"FIF cargado — {len(ev_samples)} eventos: {fif_path}")
        return ev_samples, ev_desc, sfreq
    print(f"⚠️  FIF no encontrado: {fif_path}")
    return np.array([], dtype=np.int64), [], 250.0


# ── HELPERS DE DIBUJO ─────────────────────────────────────────────────────

def _shade_trials(ax, event_samples, event_desc, trial_samples: int) -> None:
    for sample, desc in zip(event_samples, event_desc):
        color = TRIAL_COLORS.get(desc, "gray")
        ax.axvspan(sample, sample + trial_samples, color=color, alpha=0.07, linewidth=0)


def _shade_predictions(ax, samples: np.ndarray, predictions: list) -> None:
    if len(samples) < 2:
        return
    half_step = (samples[1] - samples[0]) / 2.0
    for i, pred in enumerate(predictions):
        color = PRED_COLORS.get(str(pred), "lightgray")
        ax.axvspan(samples[i] - half_step, samples[i] + half_step,
                   color=color, alpha=0.15, linewidth=0)


def _draw_event_lines(ax, event_samples, event_desc) -> None:
    """Línea vertical + etiqueta en cada evento. La etiqueta usa coordenadas mixtas
    (x en datos, y en fracción de ejes) para que funcione con cualquier escala Y."""
    unique_descs = list(dict.fromkeys(event_desc))
    cmap = plt.get_cmap("tab10")
    color_map = {d: cmap(i % cmap.N) for i, d in enumerate(unique_descs)}
    trans = ax.get_xaxis_transform()
    for sample, desc in zip(event_samples, event_desc):
        color = color_map[desc]
        ax.axvline(sample, color=color, alpha=0.6, linewidth=1.2)
        ax.text(sample, 0.97, desc, rotation=90, va="top", ha="right",
                fontsize=7, color=color, alpha=0.85, transform=trans)


def _plot_with_dots(
    ax, x, y, color, label,
    linewidth: float = 1.4,
    alpha: float = 0.9,
    linestyle: str = "-",
) -> None:
    """Dibuja la línea y superpone puntos rojos en cada punto de predicción."""
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
            linestyle=linestyle, label=label)
    ax.scatter(x, y, s=DOT_SIZE, color=DOT_COLOR, zorder=5, linewidths=0)


# ── PANELES INDIVIDUALES ──────────────────────────────────────────────────

def _draw_integrador_fuga(ax, csv_path, ev_samples, ev_desc, trial_samples):
    ax.set_title("Integrador con Fuga", fontsize=10, loc="left", fontweight="bold")
    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, f"CSV no encontrado:\n{csv_path}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=9)
        return

    df = pd.read_csv(csv_path)
    required = {"sample", "prediction", "integrator", "threshold"}
    missing = required - set(df.columns)
    if missing:
        ax.text(0.5, 0.5, f"Columnas faltantes: {missing}", ha="center",
                va="center", transform=ax.transAxes, color="red", fontsize=9)
        return

    threshold   = float(df["threshold"].iloc[0])
    samples     = df["sample"].to_numpy()
    integrator  = df["integrator"].to_numpy()
    predictions = df["prediction"].tolist()

    _shade_trials(ax, ev_samples, ev_desc, trial_samples)
    _shade_predictions(ax, samples, predictions)
    ax.fill_between(samples, integrator, 0, where=(integrator >= 0),
                    color="tab:orange", alpha=0.2, linewidth=0)
    ax.fill_between(samples, integrator, 0, where=(integrator < 0),
                    color="tab:blue",   alpha=0.2, linewidth=0)
    _plot_with_dots(ax, samples, integrator, "black", "integrador")
    ax.axhline( threshold, color="tab:orange", linestyle="--", linewidth=1.2,
               label=f"+threshold ({threshold})")
    ax.axhline(-threshold, color="tab:blue",   linestyle="--", linewidth=1.2,
               label=f"−threshold ({-threshold})")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _draw_event_lines(ax, ev_samples, ev_desc)
    ax.set_ylabel("Valor integrador")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


def _draw_score_ponderado(ax, csv_path, ev_samples, ev_desc, trial_samples):
    ax.set_title("Score Ponderado", fontsize=10, loc="left", fontweight="bold")
    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, f"CSV no encontrado:\n{csv_path}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=9)
        return

    df = pd.read_csv(csv_path)
    required = {"sample", "prediction", "score_total", "mean_score"}
    missing = required - set(df.columns)
    if missing:
        ax.text(0.5, 0.5, f"Columnas faltantes: {missing}", ha="center",
                va="center", transform=ax.transAxes, color="red", fontsize=9)
        return

    samples     = df["sample"].to_numpy()
    score_total = df["score_total"].to_numpy()
    mean_score  = df["mean_score"].to_numpy()
    predictions = df["prediction"].tolist()

    _shade_trials(ax, ev_samples, ev_desc, trial_samples)
    _shade_predictions(ax, samples, predictions)
    ax.fill_between(samples, score_total, 0, where=(score_total >= 0),
                    color="tab:orange", alpha=0.2, linewidth=0)
    ax.fill_between(samples, score_total, 0, where=(score_total < 0),
                    color="tab:blue",   alpha=0.2, linewidth=0)
    _plot_with_dots(ax, samples, score_total, "black",      "score_total")
    _plot_with_dots(ax, samples, mean_score,  "tab:purple", "mean_score",
                    linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _draw_event_lines(ax, ev_samples, ev_desc)
    ax.set_ylabel("Score de ventana")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


def _draw_media_probabilidades(ax, csv_path, ev_samples, ev_desc, trial_samples):
    ax.set_title("Media de Probabilidades de Ventana", fontsize=10, loc="left", fontweight="bold")
    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, f"CSV no encontrado:\n{csv_path}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=9)
        return

    df = pd.read_csv(csv_path)
    required = {"sample", "prediction", "p_left_raw", "p_right_raw", "mean_left", "mean_right"}
    missing = required - set(df.columns)
    if missing:
        ax.text(0.5, 0.5, f"Columnas faltantes: {missing}", ha="center",
                va="center", transform=ax.transAxes, color="red", fontsize=9)
        return

    samples     = df["sample"].to_numpy()
    p_left_raw  = df["p_left_raw"].to_numpy()
    p_right_raw = df["p_right_raw"].to_numpy()
    mean_left   = df["mean_left"].to_numpy()
    mean_right  = df["mean_right"].to_numpy()
    predictions = df["prediction"].tolist()

    _shade_trials(ax, ev_samples, ev_desc, trial_samples)
    _shade_predictions(ax, samples, predictions)
    ax.plot(samples, p_left_raw,  color="tab:blue",   alpha=0.25, linewidth=0.8, linestyle="--",
            label="p_left raw")
    ax.plot(samples, p_right_raw, color="tab:orange", alpha=0.25, linewidth=0.8, linestyle="--",
            label="p_right raw")
    _plot_with_dots(ax, samples, mean_left,  "tab:blue",   "mean_left")
    _plot_with_dots(ax, samples, mean_right, "tab:orange", "mean_right")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _draw_event_lines(ax, ev_samples, ev_desc)
    ax.set_ylabel("Probabilidad media")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


def _draw_exponential_smoothing(ax, csv_path, ev_samples, ev_desc, trial_samples):
    ax.set_title("Exponential Smoothing", fontsize=10, loc="left", fontweight="bold")
    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, f"CSV no encontrado:\n{csv_path}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=9)
        return

    df = pd.read_csv(csv_path)
    required = {"sample", "prediction", "p_left_raw", "p_right_raw", "p_left_smooth", "p_right_smooth"}
    missing = required - set(df.columns)
    if missing:
        ax.text(0.5, 0.5, f"Columnas faltantes: {missing}", ha="center",
                va="center", transform=ax.transAxes, color="red", fontsize=9)
        return

    alpha_val = float(df["alpha"].iloc[0])     if "alpha"     in df.columns else None
    threshold = float(df["threshold"].iloc[0]) if "threshold" in df.columns else None

    samples        = df["sample"].to_numpy()
    p_left_raw     = df["p_left_raw"].to_numpy()
    p_right_raw    = df["p_right_raw"].to_numpy()
    p_left_smooth  = df["p_left_smooth"].to_numpy()
    p_right_smooth = df["p_right_smooth"].to_numpy()
    predictions    = df["prediction"].tolist()

    extra = "  |"
    if alpha_val is not None:
        extra += f"  α={alpha_val}"
    if threshold is not None:
        extra += f"  threshold={threshold}"
    if extra.strip("|").strip():
        ax.set_title(f"Exponential Smoothing{extra}", fontsize=10, loc="left", fontweight="bold")

    _shade_trials(ax, ev_samples, ev_desc, trial_samples)
    _shade_predictions(ax, samples, predictions)
    ax.plot(samples, p_left_raw,  color="tab:blue",   alpha=0.25, linewidth=0.8, linestyle="--",
            label="p_left raw")
    ax.plot(samples, p_right_raw, color="tab:orange", alpha=0.25, linewidth=0.8, linestyle="--",
            label="p_right raw")
    _plot_with_dots(ax, samples, p_left_smooth,  "tab:blue",   "left smooth")
    _plot_with_dots(ax, samples, p_right_smooth, "tab:orange", "right smooth")
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
                   label=f"threshold ({threshold})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _draw_event_lines(ax, ev_samples, ev_desc)
    ax.set_ylabel("Probabilidad suavizada")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


# ── FUNCIÓN PRINCIPAL ─────────────────────────────────────────────────────

def visor_comparativo() -> None:
    ev_fuga,  desc_fuga,  sfreq_fuga  = _load_fif(FIF_FUGA)
    ev_score, desc_score, sfreq_score = _load_fif(FIF_SCORE)
    ev_media, desc_media, sfreq_media = _load_fif(FIF_MEDIA)
    ev_es,    desc_es,    sfreq_es    = _load_fif(FIF_ES)

    # Para la leyenda global, usamos los eventos del primer FIF disponible
    all_ev_desc = desc_fuga or desc_score or desc_media or desc_es

    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
    fig.suptitle("Comparativa de Métodos de Post-Procesado RT", fontsize=13, fontweight="bold")

    _draw_integrador_fuga      (axes[0], CSV_FUGA,  ev_fuga,  desc_fuga,  int(TRIAL_DURATION_S * sfreq_fuga))
    _draw_score_ponderado      (axes[1], CSV_SCORE, ev_score, desc_score, int(TRIAL_DURATION_S * sfreq_score))
    _draw_media_probabilidades (axes[2], CSV_MEDIA, ev_media, desc_media, int(TRIAL_DURATION_S * sfreq_media))
    _draw_exponential_smoothing(axes[3], CSV_ES,    ev_es,    desc_es,    int(TRIAL_DURATION_S * sfreq_es))

    axes[3].set_xlabel("Número de muestra")

    # Leyenda global en la parte inferior
    pred_patches = [
        mpatches.Patch(color=PRED_COLORS["left_hand"],  alpha=0.3, label="pred: left_hand"),
        mpatches.Patch(color=PRED_COLORS["right_hand"], alpha=0.3, label="pred: right_hand"),
        mpatches.Patch(color=PRED_COLORS["rest"],       alpha=0.3, label="pred: rest"),
    ]
    trial_patches = [
        mpatches.Patch(color=TRIAL_COLORS.get(cls, "gray"), alpha=0.15, label=f"trial: {cls}")
        for cls in dict.fromkeys(all_ev_desc)
        if cls in TRIAL_COLORS
    ]
    dot_patch = mpatches.Patch(color=DOT_COLOR, label="punto de predicción")
    fig.legend(
        handles=pred_patches + trial_patches + [dot_patch],
        loc="lower center",
        ncol=len(pred_patches) + len(trial_patches) + 1,
        fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


if __name__ == "__main__":
    visor_comparativo()
