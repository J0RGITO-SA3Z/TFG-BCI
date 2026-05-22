"""
Detects and visualises EMG (muscle) and EOG (blink) artefacts from a raw FIF file.
Outputs two PNG figures: artifact_emg.png and artifact_eog.png
Usage: python plot_artifacts.py [path_to_raw.fif]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
from scipy.signal import hilbert, find_peaks

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_RAW = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "recordings", "piloto", "suj1", "suj1_1_raw.fif"
)
RAW_PATH = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW)
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Cargando: {RAW_PATH}")
raw = mne.io.read_raw_fif(RAW_PATH, preload=True, verbose=False)
sfreq = raw.info["sfreq"]
eeg_idx = mne.pick_types(raw.info, eeg=True, exclude="bads")
eeg_chs  = [raw.ch_names[i] for i in eeg_idx]
print(f"  Canales EEG: {eeg_chs}")
print(f"  Duración: {raw.times[-1]:.1f}s | Fs: {sfreq:.0f} Hz")

# BrainAccess saves raw ADC values (cal=1, no V/µV conversion).
# We demean per channel so all subsequent operations work in relative ADC units.
raw_data_all = raw.get_data(picks=eeg_idx)         # (n_ch, n_times)  ADC counts
ch_means     = raw_data_all.mean(axis=1, keepdims=True)
raw_demeaned = raw_data_all - ch_means              # zero-mean per channel

# ── Channel selection ──────────────────────────────────────────────────────────
MAX_CH   = 8
disp_chs = eeg_chs[:MAX_CH]

# Frontal channels for blink (F3/F4/Fz are best available without Fp1/Fp2)
FRONTAL_PREF = ["F3", "F4", "Fz", "FC1", "FC2"]
frontal_chs  = [ch for ch in FRONTAL_PREF if ch in eeg_chs] or eeg_chs[:2]
frontal_idx  = [eeg_chs.index(ch) for ch in frontal_chs]
print(f"  Canales frontales para EOG: {frontal_chs}")

# ══════════════════════════════════════════════════════════════════════════════
#  EMG ARTEFACT DETECTION  (z-score on high-freq power envelope)
# ══════════════════════════════════════════════════════════════════════════════
print("Detectando artefactos musculares (EMG)...")

def _detect_muscle(demeaned, sfreq, raw, eeg_chs, lo=60, hi=None, z_thresh=4.0, min_dur=0.05):
    hi = hi or min(115.0, sfreq / 2 - 5)
    raw_hf = raw.copy().pick(eeg_chs).filter(lo, hi, verbose=False)
    data   = raw_hf.get_data() - raw_hf.get_data().mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(data, axis=1)).mean(axis=0)
    z = (envelope - envelope.mean()) / envelope.std()
    above = (z > z_thresh).astype(int)
    events = []
    in_seg, seg_start = False, 0
    for i, v in enumerate(above):
        if v and not in_seg:
            in_seg, seg_start = True, i
        elif not v and in_seg:
            dur = (i - seg_start) / sfreq
            if dur >= min_dur:
                events.append((seg_start / sfreq, dur, envelope[seg_start:i].max()))
            in_seg = False
    if in_seg:
        dur = (len(above) - seg_start) / sfreq
        if dur >= min_dur:
            events.append((seg_start / sfreq, dur, envelope[seg_start:].max()))
    return events

hi_emg = min(115.0, sfreq / 2 - 5)
muscle_events = _detect_muscle(raw_demeaned, sfreq, raw, eeg_chs, z_thresh=4.0)
if not muscle_events:
    muscle_events = _detect_muscle(raw_demeaned, sfreq, raw, eeg_chs, z_thresh=2.5)
# Sort by peak envelope power → pick strongest artifact
muscle_events.sort(key=lambda x: x[2], reverse=True)
print(f"  Encontrados {len(muscle_events)} segmentos EMG")

# ══════════════════════════════════════════════════════════════════════════════
#  EOG / BLINK DETECTION  (frontal channels, 0.5–15 Hz, peak amplitude)
# ══════════════════════════════════════════════════════════════════════════════
print("Detectando parpadeos (EOG)...")

# Filter frontal channels 0.5-15 Hz to preserve sharp blink transient
raw_eog = raw.copy().pick(frontal_chs).filter(0.5, 15.0, verbose=False)
fp_data = raw_eog.get_data()                           # already calibrated (ADC units)
fp_data -= fp_data.mean(axis=1, keepdims=True)         # demean
fp_mean  = fp_data.mean(axis=0)                        # average frontal signal

# Use BOTH positive and negative extremes (blinks can be + or – depending on ref)
fp_abs   = np.abs(fp_mean)
thresh   = fp_abs.mean() + 3.5 * fp_abs.std()
peaks_p, props_p = find_peaks( fp_mean,  height=thresh, distance=int(0.4 * sfreq), prominence=thresh * 0.5)
peaks_n, props_n = find_peaks(-fp_mean,  height=thresh, distance=int(0.4 * sfreq), prominence=thresh * 0.5)

# Combine, keep track of amplitude for ranking
blink_candidates = (
    [(p, fp_abs[p]) for p in peaks_p] +
    [(p, fp_abs[p]) for p in peaks_n]
)
blink_candidates.sort(key=lambda x: x[1], reverse=True)   # largest first
blink_times = [p / sfreq for p, _ in blink_candidates]
print(f"  Encontrados {len(blink_times)} parpadeos candidatos")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
HALF_WIN = 1.5    # seconds each side

STYLE = {
    "emg": {"color": "#E53935", "label": "artefacto EMG"},
    "eog": {"color": "#1E88E5", "label": "parpadeo EOG"},
}


def _get_window(raw, eeg_chs, ch_means, disp_chs, t_center):
    """Return (times, demeaned_data) for display channels in ±HALF_WIN window."""
    t0 = max(0.0, t_center - HALF_WIN)
    t1 = min(raw.times[-1], t_center + HALF_WIN)
    si0, si1 = int(t0 * sfreq), int(t1 * sfreq)
    picks = [eeg_chs.index(ch) for ch in disp_chs]
    data  = raw.get_data(picks=picks, start=si0, stop=si1)
    data -= ch_means[picks, :][:, :si1 - si0]     # subtract per-channel mean (slice of same length)
    # re-demean within window to remove slow drift
    data -= data.mean(axis=1, keepdims=True)
    times = np.linspace(t0, t1, data.shape[1])
    return times, data


def _offset_plot(ax, times, data, ch_names, art_start, art_end, color):
    """Stacked EEG trace plot. Auto-scales slot height from data."""
    n_ch = len(ch_names)
    # Compute a robust per-channel display scale: clip at 99th percentile
    scales = np.percentile(np.abs(data), 99, axis=1)           # (n_ch,)
    scales = np.where(scales < 1e-3, 1e-3, scales)
    slot   = float(np.median(scales) * 2.5)                    # spacing in ADC units

    offsets = np.arange(n_ch, 0, -1) * slot                    # top channel at highest y

    ax.axvspan(art_start, art_end, alpha=0.18, color=color, zorder=0)
    ax.axvline(art_start, color=color, lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.axvline(art_end,   color=color, lw=1.0, ls="--", alpha=0.7, zorder=1)

    for i, (ch, row, offset) in enumerate(zip(ch_names, data, offsets)):
        clipped = np.clip(row, -slot / 2, slot / 2)
        ax.plot(times, clipped + offset, lw=0.9, color="#333333", alpha=0.9, zorder=2)
        ax.text(times[0] - 0.03 * (times[-1] - times[0]), offset, ch,
                ha="right", va="center", fontsize=8.5, color="#333333")

    # Scale bar
    bar_x  = times[-1] - 0.04 * (times[-1] - times[0])
    bar_y  = offsets[-1] - slot * 0.6
    ax.plot([bar_x, bar_x], [bar_y, bar_y + slot / 2],
            color="#666666", lw=2.5, solid_capstyle="butt", zorder=3)
    ax.text(bar_x + 0.015 * (times[-1] - times[0]), bar_y + slot / 4,
            "50 µV*", va="center", fontsize=7.5, color="#666666")

    ax.set_yticks([])
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(offsets[-1] - slot, offsets[0] + slot)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def plot_artifact_window(raw, eeg_chs, ch_means, center, art_start, art_end, kind, out_path):
    times, data = _get_window(raw, eeg_chs, ch_means, disp_chs, center)

    style = STYLE[kind]
    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 8),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.10},
    )

    # ── Top panel: stacked EEG ─────────────────────────────────────────────
    _offset_plot(axes[0], times, data, disp_chs, art_start, art_end, style["color"])
    title_kind = "muscular (EMG)" if kind == "emg" else "ocular — parpadeo (EOG)"
    axes[0].set_title(
        f"Artefacto {title_kind}  —  t ≈ {center:.2f} s",
        fontsize=13, fontweight="bold", pad=10,
    )

    # ── Bottom panel: band-filtered artefact signature ────────────────────
    si0 = int(max(0.0, center - HALF_WIN) * sfreq)
    si1 = int(min(raw.times[-1], center + HALF_WIN) * sfreq)

    if kind == "emg":
        filt = raw.copy().pick(disp_chs).filter(60, hi_emg, verbose=False)
        fdata = filt.get_data(start=si0, stop=si1)
        fdata -= fdata.mean(axis=1, keepdims=True)
        band_label = f"Filtrado 60–{hi_emg:.0f} Hz (banda muscular)"
        for row in fdata:
            axes[1].plot(times, row, lw=0.8, alpha=0.6, color=style["color"])
    else:
        filt = raw.copy().pick(frontal_chs).filter(0.5, 15.0, verbose=False)
        fdata = filt.get_data(start=si0, stop=si1)
        fdata -= fdata.mean(axis=1, keepdims=True)
        band_label = f"Filtrado 0.5–15 Hz  [{', '.join(frontal_chs)}]  (parpadeo)"
        for i, row in enumerate(fdata):
            ax_color = plt.cm.Blues(0.5 + 0.4 * i / max(len(fdata) - 1, 1))
            axes[1].plot(times, row, lw=1.2, alpha=0.85, color=ax_color,
                         label=frontal_chs[i])
        axes[1].legend(fontsize=8, loc="upper right", framealpha=0.6)

    axes[1].axvspan(art_start, art_end, alpha=0.20, color=style["color"])
    axes[1].axvline(center, color=style["color"], lw=1.2, ls=":", alpha=0.8)
    axes[1].set_ylabel(band_label, fontsize=8)
    axes[1].set_xlabel("Tiempo (s)", fontsize=10)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}")
    )
    for sp in ("top", "right"):
        axes[1].spines[sp].set_visible(False)

    patch = mpatches.Patch(color=style["color"], alpha=0.5, label=style["label"])
    axes[0].legend(handles=[patch], loc="upper right", fontsize=9, framealpha=0.7)
    axes[0].text(0.01, 0.01, "* escala orientativa (unidades ADC relativas)",
                 transform=axes[0].transAxes, fontsize=7, color="#888888", va="bottom")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Guardado: {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# EMG ──────────────────────────────────────────────────────────────────────────
if muscle_events:
    onset, dur, _ = muscle_events[0]
    center = onset + dur / 2
    plot_artifact_window(
        raw, eeg_chs, ch_means, center, onset, onset + dur, "emg",
        os.path.join(OUT_DIR, "artifact_emg.png"),
    )
else:
    print("  No se encontraron artefactos musculares.")

# EOG ──────────────────────────────────────────────────────────────────────────
if blink_times:
    t = blink_times[0]
    plot_artifact_window(
        raw, eeg_chs, ch_means, t, t - 0.12, t + 0.30, "eog",
        os.path.join(OUT_DIR, "artifact_eog.png"),
    )
else:
    print("  No se encontraron parpadeos.")

print("Listo.")
