"""
Interpolación de canales faltantes mediante spline esférico de MNE.

Misma interfaz que SpatialInterpolator y ZeroInterpolator.
"""

from typing import List, Optional

import numpy as np
import mne

from components.EpochProcessing.EpochProcessor import EpochProcessor

import os, sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

from components.pretrainedModels.MiRepNet.utils.channel_list import (
    channel_positions as DEFAULT_CHANNEL_POSITIONS,
    use_channels_names as DEFAULT_TARGET_CHANNELS,
)

_MONTAGE = mne.channels.make_standard_montage("standard_1005")
_MONTAGE_MAP = {ch.upper(): ch for ch in _MONTAGE.ch_names}


def _spherical_spline_interpolation(
    data: np.ndarray,
    target_channels: List[str],
    actual_channels: List[str],
    sfreq: float = 250.0,
) -> np.ndarray:
    """
    Devuelve (B, C_target, T) interpolando los canales ausentes con spline esférico de MNE.
    Los canales presentes se copian directamente; solo se interpolan los faltantes.
    """
    B, _, T = data.shape
    actual_upper = [ch.upper() for ch in actual_channels]
    target_upper = [ch.upper() for ch in target_channels]

    # Construir array con los canales target (presentes copiados, faltantes a 0 → se interpolarán)
    out = np.zeros((B, len(target_upper), T), dtype=data.dtype)
    missing_mne = []
    for dst_idx, ch in enumerate(target_upper):
        if ch in actual_upper:
            out[:, dst_idx, :] = data[:, actual_upper.index(ch), :]
        else:
            missing_mne.append(_MONTAGE_MAP.get(ch, ch))

    if not missing_mne:
        return out

    mne_ch_names = [_MONTAGE_MAP.get(ch, ch) for ch in target_upper]
    info = mne.create_info(ch_names=mne_ch_names, sfreq=sfreq, ch_types="eeg", verbose=False)

    events = np.column_stack([
        np.arange(B) * (T + 1),
        np.zeros(B, dtype=int),
        np.ones(B, dtype=int),
    ])

    epochs_tmp = mne.EpochsArray(out, info=info, events=events, verbose=False)
    epochs_tmp.set_montage(_MONTAGE, on_missing="ignore", verbose=False)
    epochs_tmp.info["bads"] = missing_mne
    epochs_tmp.interpolate_bads(reset_bads=True, verbose=False)

    return epochs_tmp.get_data()


class SphericalSplineInterpolator(EpochProcessor):
    """
    Rellena los canales faltantes usando interpolación de spline esférico de MNE
    y reordena a ``target_channels``.
    """

    def __init__(
        self,
        target_channels: Optional[List[str]] = None,
        actual_channel_positions: Optional[List[str]] = None,
    ) -> None:
        self.target_channels = target_channels if target_channels is not None else DEFAULT_TARGET_CHANNELS
        self.channel_positions = DEFAULT_CHANNEL_POSITIONS

        if actual_channel_positions is not None:
            self.actual_channel_positions = [ch.upper() for ch in actual_channel_positions]
        else:
            self.actual_channel_positions = None

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        actual_channels = [ch.upper() for ch in epochs.ch_names]
        data = epochs.get_data()  # (B, C_actual, T)
        sfreq = epochs.info["sfreq"]

        interpolated = _spherical_spline_interpolation(
            data, self.target_channels, actual_channels, sfreq=sfreq
        )

        new_channels = [_MONTAGE_MAP.get(ch.upper(), ch) for ch in self.target_channels]
        return self._to_epochs(interpolated, epochs, new_channels)

    def process_np(self, X: np.ndarray, Y: np.ndarray | None = None):
        if self.actual_channel_positions is None:
            raise Exception("No se han proporcionado las posiciones de los canales reales.")

        interpolated = _spherical_spline_interpolation(
            X, self.target_channels, self.actual_channel_positions
        )
        return interpolated, Y

    def __repr__(self) -> str:
        return f"SphericalSplineInterpolator(n_target={len(self.target_channels)})"
