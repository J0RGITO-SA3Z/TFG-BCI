"""
Rellena los canales faltantes con ceros en lugar de interpolarlos.

Misma interfaz que SpatialInterpolator, útil como baseline de comparación.
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


def _pad_with_zeros(
    data: np.ndarray,
    target_channels: List[str],
    actual_channels: List[str],
) -> np.ndarray:
    """Devuelve (B, C_target, T) colocando ceros en los canales ausentes."""
    B, _, T = data.shape
    actual_upper = [ch.upper() for ch in actual_channels]
    target_upper = [ch.upper() for ch in target_channels]

    out = np.zeros((B, len(target_upper), T), dtype=data.dtype)
    for dst_idx, ch in enumerate(target_upper):
        if ch in actual_upper:
            src_idx = actual_upper.index(ch)
            out[:, dst_idx, :] = data[:, src_idx, :]
    return out


class ZeroInterpolator(EpochProcessor):
    """
    Rellena los canales faltantes con ceros y reordena a ``target_channels``.
    Sirve como baseline frente a SpatialInterpolator.
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

        padded = _pad_with_zeros(data, self.target_channels, actual_channels)

        new_channels = [self._validar_nombre_electrodo(ch) for ch in self.target_channels]
        return self._to_epochs(padded, epochs, new_channels)

    def process_np(self, X: np.ndarray, Y: np.ndarray | None = None):
        if self.actual_channel_positions is None:
            raise Exception("No se han proporcionado las posiciones de los canales reales.")

        padded = _pad_with_zeros(X, self.target_channels, self.actual_channel_positions)
        return padded, Y

    def _validar_nombre_electrodo(self, nombre: str) -> str | None:
        montage = mne.channels.make_standard_montage("standard_1005")
        mapa = {ch.upper(): ch for ch in montage.ch_names}
        return mapa.get(nombre.strip().upper(), None)

    def __repr__(self) -> str:
        return f"ZeroInterpolator(n_target={len(self.target_channels)})"
