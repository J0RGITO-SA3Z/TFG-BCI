"""
Interpolación espacial de canales sobre Epochs.

Rellena canales faltantes interpolando por distancia inversa (IDW)
a partir de las posiciones 2-D de los electrodos, reutilizando la lógica
de ``pad_missing_channels_diff`` de MIRepNet.
"""

from typing import Dict, List, Optional, Tuple

import os, sys
import numpy as np
import mne

from .EpochProcessor import EpochProcessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # sube desde src/epoch_processing -> src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrainedModels.MiRepNet.utils import (
    channel_positions as DEFAULT_CHANNEL_POSITIONS,
    use_channels_names as DEFAULT_TARGET_CHANNELS,
)
from pretrainedModels.MiRepNet.utils.utils import pad_missing_channels_diff

class SpatialInterpolator(EpochProcessor):
    """
    Interpola / rellena canales para que los Epochs resultantes tengan
    exactamente ``target_channels`` canales en ese orden.
    """

    def __init__(
        self,
        target_channels: Optional[List[str]] = None,
        channel_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.target_channels = target_channels if target_channels is not None else DEFAULT_TARGET_CHANNELS
        self.channel_positions = channel_positions if channel_positions is not None else DEFAULT_CHANNEL_POSITIONS

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Recibe ``mne.Epochs`` y devuelve nuevos ``mne.Epochs`` con los
        canales interpolados / reordenados a ``target_channels``.
        """
        actual_channels = [ch.upper() for ch in epochs.ch_names]
        data = epochs.get_data()  # (B, C_actual, T)

        # Reutilizamos la implementación central de MIRepNet
        interpolated = pad_missing_channels_diff(
            data, self.target_channels, actual_channels,
        )  # (B, C_target, T)

        return self._to_epochs(interpolated, epochs)

    def __repr__(self) -> str:
        return f"SpatialInterpolator(n_target={len(self.target_channels)})"
