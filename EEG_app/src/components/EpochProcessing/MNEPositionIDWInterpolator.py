"""
Interpolación IDW de canales faltantes usando las posiciones 3D de MNE (standard_1005).

Misma lógica que SpatialInterpolator pero con coordenadas esféricas de MNE
en lugar de las posiciones 2D planas de MiRepNet.
"""

from typing import List, Optional

import numpy as np
from scipy.spatial.distance import cdist
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
    use_channels_names as DEFAULT_TARGET_CHANNELS,
)

# Extraemos las posiciones 3D del montaje estándar de MNE una sola vez
_MONTAGE = mne.channels.make_standard_montage("standard_1005")
_MNE_POSITIONS: dict[str, np.ndarray] = {
    ch.upper(): pos
    for ch, pos in zip(_MONTAGE.ch_names, _MONTAGE.get_positions()["ch_pos"].values())
}


def _idw_with_mne_positions(
    data: np.ndarray,
    target_channels: List[str],
    actual_channels: List[str],
) -> np.ndarray:
    """
    Devuelve (B, C_target, T) usando IDW con posiciones 3D de MNE.
    Los canales presentes se copian con peso 1; los faltantes se interpolan.
    """
    B, C, T = data.shape
    actual_upper = [ch.upper() for ch in actual_channels]
    target_upper = [ch.upper() for ch in target_channels]

    existing_pos = np.array([_MNE_POSITIONS[ch] for ch in actual_upper])
    target_pos   = np.array([_MNE_POSITIONS[ch] for ch in target_upper])

    W = np.zeros((len(target_upper), C))
    for i, (tch, pos) in enumerate(zip(target_upper, target_pos)):
        if tch in actual_upper:
            W[i, actual_upper.index(tch)] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1.0 / (dist + 1e-6)
            weights /= weights.sum()
            W[i] = weights

    padded = np.zeros((B, len(target_upper), T), dtype=data.dtype)
    for b in range(B):
        padded[b] = W @ data[b]
    return padded


class MNEPositionIDWInterpolator(EpochProcessor):
    """
    IDW sobre posiciones 3D del montaje standard_1005 de MNE.
    Alternativa a SpatialInterpolator, que usa posiciones 2D de MiRepNet.
    """

    def __init__(
        self,
        target_channels: Optional[List[str]] = None,
        actual_channel_positions: Optional[List[str]] = None,
    ) -> None:
        self.target_channels = target_channels if target_channels is not None else DEFAULT_TARGET_CHANNELS

        missing = [ch for ch in self.target_channels if ch.upper() not in _MNE_POSITIONS]
        if missing:
            raise ValueError(f"Canales sin posición en MNE standard_1005: {missing}")

        if actual_channel_positions is not None:
            self.actual_channel_positions = [ch.upper() for ch in actual_channel_positions]
            missing_act = [ch for ch in self.actual_channel_positions if ch not in _MNE_POSITIONS]
            if missing_act:
                raise ValueError(f"Canales reales sin posición en MNE standard_1005: {missing_act}")
        else:
            self.actual_channel_positions = None

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        actual_channels = [ch.upper() for ch in epochs.ch_names]
        data = epochs.get_data()  # (B, C_actual, T)

        interpolated = _idw_with_mne_positions(data, self.target_channels, actual_channels)

        montage_map = {ch.upper(): ch for ch in _MONTAGE.ch_names}
        new_channels = [montage_map.get(ch.upper(), ch) for ch in self.target_channels]
        return self._to_epochs(interpolated, epochs, new_channels)

    def process_np(self, X: np.ndarray, Y: np.ndarray | None = None):
        if self.actual_channel_positions is None:
            raise Exception("No se han proporcionado las posiciones de los canales reales.")

        interpolated = _idw_with_mne_positions(X, self.target_channels, self.actual_channel_positions)
        return interpolated, Y

    def __repr__(self) -> str:
        return f"MNEPositionIDWInterpolator(n_target={len(self.target_channels)})"
