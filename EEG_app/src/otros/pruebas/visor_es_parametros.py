"""
Visor de comparación de parámetros de Suavizado Exponencial.
Pide por consola el número de gráficas y los parámetros (alpha, threshold) de cada una,
y genera una única ventana con todas las gráficas apiladas.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

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
PRED_COLORS = {
    "left_hand":  "tab:blue",
    "right_hand": "tab:orange",
    "rest":       "tab:gray",
}
DOT_SIZE      = 20
_EVENT_HIDDEN = {"feet", "cross"}


# ── ES: recomputar ────────────────────────────────────────────────────────

def _recompute_smooth(p_raw: np.ndarray, alpha: float) -> np.ndarray:
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
    preds = []
    for l, r in zip(p_left_smooth, p_right_smooth):
        if r > threshold:
            preds.append("right_hand")
        elif l > threshold:
            preds.append("left_hand")
        else:
            preds.append("rest")
    return preds


# ── FIF ───────────────────────────────────────────────────────────────────

def _build_sample_track(raw: mne.io.BaseRaw) -> np.ndarray:
    if "Sample" in raw.ch_names:
        return np.asarray(raw.get_data(picks=["Sample"])[0], dtype=np.int64)
    return np.arange(raw.first_samp, raw.first_samp + raw.n_times, dtype=np.int64)


def _load_fif(fif_path: str) -> tuple[np.ndarray, list[str]]:
    if fif_path and os.path.exists(fif_path):
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        track = _build_sample_track(raw)
        if len(raw.annotations) == 0:
            return np.array([], dtype=np.int64), []
        idx = raw.time_as_index(raw.annotations.onset, use_rounding=True)
        idx = np.clip(np.asarray(idx, dtype=np.int64), 0, len(track) - 1)
        ev_samples = track[idx]
        ev_desc = [ANNOTATION_MAP.get(d, d) for d in raw.annotations.description]
        print(f"FIF cargado — {len(ev_samples)} eventos.")
        return ev_samples, ev_desc
    print(f"⚠️  FIF no encontrado (se omiten eventos): {fif_path}")
    return np.array([], dtype=np.int64), []


# ── Helpers de dibujo ─────────────────────────────────────────────────────

def _scatter_pred_dots(ax, samples: np.ndarray, values: np.ndarray, predictions: list) -> None:
    pred_arr = np.array([str(p) for p in predictions])
    for pred in dict.fromkeys(str(p) for p in predictions):
        mask = pred_arr == pred
        color = PRED_COLORS.get(pred, "lightgray")
        ax.scatter(samples[mask], values[mask], s=DOT_SIZE, color=color, zorder=5, linewidths=0)


def _draw_event_lines(ax, ev_samples, ev_desc) -> None:
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


def _draw_panel(
    ax, samples, p_right_raw, p_right_smooth, predictions,
    ev_samples, ev_desc, alpha, threshold,
) -> None:
    ax.set_facecolor("white")
    low = round(1.0 - threshold, 4)

    ax.scatter(samples, p_right_raw, s=DOT_SIZE * 0.4, color="red", alpha=0.85,
               zorder=2, label="p_right raw", linewidths=0)
    ax.plot(samples, p_right_smooth, color="tab:orange", linewidth=1.4, alpha=0.9,
            label="right smooth")
    _scatter_pred_dots(ax, samples, p_right_smooth, predictions)

    ax.axhline(threshold, color="tab:orange", linestyle="--", linewidth=2.2, alpha=1.0,
               label=f"thr right_hand ({threshold})")
    ax.axhline(low,       color="tab:blue",   linestyle="--", linewidth=2.2, alpha=1.0,
               label=f"thr left_hand  ({low})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    _draw_event_lines(ax, ev_samples, ev_desc)

    ax.set_ylabel("p right_hand")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"α = {alpha}  |  threshold = {threshold}", fontsize=9,
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)


# ── Input por consola ─────────────────────────────────────────────────────

def _ask_int(prompt: str, min_val: int = 1) -> int:
    while True:
        try:
            val = int(input(prompt))
            if val >= min_val:
                return val
            print(f"  Introduce un número >= {min_val}.")
        except ValueError:
            print("  Valor no válido.")


def _ask_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Debe estar entre {lo} y {hi}.")
        except ValueError:
            print("  Valor no válido.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV no encontrado: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    missing = {"sample", "p_left_raw", "p_right_raw"} - set(df.columns)
    if missing:
        print(f"❌ Columnas faltantes: {missing}")
        return

    samples     = df["sample"].to_numpy(dtype=np.int64)
    p_left_raw  = df["p_left_raw"].to_numpy(dtype=float)
    p_right_raw = df["p_right_raw"].to_numpy(dtype=float)
    ev_samples, ev_desc = _load_fif(FIF_PATH)

    print()
    n = _ask_int("Número de gráficas: ")

    params: list[tuple[float, float]] = []
    for i in range(n):
        print(f"\n── Gráfica {i + 1} ──")
        alpha     = _ask_float("  α (peso pasado, 0.01–0.99): ", 0.01, 0.99)
        threshold = _ask_float("  threshold (0.50–0.99):       ", 0.50, 0.99)
        params.append((alpha, threshold))

    fig, axes = plt.subplots(n, 1, figsize=(17, 4 * n), sharex=True)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    fig.suptitle("Comparación parámetros Suavizado Exponencial",
                 fontsize=13, fontweight="bold")

    for ax, (alpha, threshold) in zip(axes, params):
        p_left_smooth  = _recompute_smooth(p_left_raw,  alpha)
        p_right_smooth = _recompute_smooth(p_right_raw, alpha)
        preds = _recompute_predictions(p_left_smooth, p_right_smooth, threshold)
        _draw_panel(ax, samples, p_right_raw, p_right_smooth, preds,
                    ev_samples, ev_desc, alpha, threshold)

    axes[-1].set_xlabel("Número de muestra")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
