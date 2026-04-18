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

from components.EpochProcessing.EpochProcessor import EpochProcessor

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
from components.pretrainedModels.MiRepNet.utils.utils import pad_missing_channels_diff

class SpatialInterpolator(EpochProcessor):
    """
    Interpola / rellena canales para que los Epochs resultantes tengan
    exactamente ``target_channels`` canales en ese orden.
    """

    def __init__(
        self,
        target_channels: Optional[List[str]] = None,
        actual_channel_positions: Optional[List[str]] = None,
    ) -> None:
        self.target_channels = target_channels if target_channels is not None else DEFAULT_TARGET_CHANNELS
        self.channel_positions =  DEFAULT_CHANNEL_POSITIONS

        if actual_channel_positions is not None:
            self.actual_channel_positions = [ch.upper() for ch in actual_channel_positions]
            for ch in self.actual_channel_positions:
                if ch.upper() not in self.channel_positions:
                    raise Exception("Nombre de canal no valido")
        else:
            self.actual_channel_positions = None

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

        newMneChannels = [self.validar_nombre_electrodo(ch) for ch in self.target_channels]

        return self._to_epochs(interpolated,epochs,newMneChannels)
    
    def process_np(self, X: np.ndarray, Y: np.ndarray | None = None):
        if self.actual_channel_positions is None:
            raise Exception("No se han proporcionado las posiciones de los canales reales, no se puede interpolar.")
        
        interpolated = pad_missing_channels_diff(
            X, self.target_channels, self.actual_channel_positions,
        )

        return interpolated, Y
    
    # ------------------------------------------------------------------
    # Funciones auxiliares
    # ------------------------------------------------------------------
    def validar_nombre_electrodo(self,nombre):
        montage = mne.channels.make_standard_montage("standard_1005")
        nombres_mne = montage.ch_names

        nombre = nombre.strip().upper()
        mapa = {ch.upper(): ch for ch in nombres_mne}

        return mapa.get(nombre, None)

    def __repr__(self) -> str:
        return f"SpatialInterpolator(n_target={len(self.target_channels)})"
