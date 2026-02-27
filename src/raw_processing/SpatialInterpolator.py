"""
Procesador: interpolación espacial de canales EEG sobre MNE Raw.

Interpola (o copia directamente) los canales del Raw de entrada para
producir un nuevo Raw con exactamente los canales objetivo (target),
usando Inverse Distance Weighting (IDW) para los canales ausentes —
exactamente la misma lógica que ``pad_missing_channels_diff`` de MIRepNet.
"""

from __future__ import annotations

import numpy as np
import mne
from scipy.spatial.distance import cdist

from .Processor import RawProcessor
from pretrainedModels.MiRepNet.utils import channel_positions, use_channels_names


class SpatialInterpolator(RawProcessor):
    """
    Interpola/re-mapea los canales del Raw de entrada a un conjunto de canales
    objetivo usando las posiciones 2-D de los electrodos.

    * Si un canal objetivo ya existe en el Raw, se copia tal cual.
    * Si no existe, se genera como media ponderada por inversa de la distancia
      (IDW) de todos los canales existentes.

    Args:
        target_channels: Lista de nombres de canales objetivo (en mayúsculas).
                         Por defecto usa ``use_channels_names`` de MIRepNet
                         (45 canales).
        positions:       Diccionario ``{nombre: (x, y)}`` con las posiciones
                         2-D de los electrodos. Por defecto usa
                         ``channel_positions`` de MIRepNet.
    """

    def __init__(
        self,
        target_channels: list[str] | None = None,
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.target_channels = target_channels or list(use_channels_names)
        self.positions = positions or dict(channel_positions)

    # ------------------------------------------------------------------

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()

        actual_channels = [ch.upper() for ch in raw.ch_names]
        data = raw.get_data()  # (C_actual, T)
        C_actual, T = data.shape

        num_target = len(self.target_channels)

        # --- Construir matriz de pesos W  (num_target × C_actual) ---
        existing_pos = np.array(
            [self.positions[ch] for ch in actual_channels]
        )
        W = np.zeros((num_target, C_actual))

        for i, target_ch in enumerate(self.target_channels):
            if target_ch in actual_channels:
                src_idx = actual_channels.index(target_ch)
                W[i, src_idx] = 1.0
            else:
                target_pos = np.array(self.positions[target_ch]).reshape(1, -1)
                dist = cdist(target_pos, existing_pos)[0]
                weights = 1.0 / (dist + 1e-6)
                weights /= weights.sum()
                W[i] = weights

        # --- Aplicar interpolación ---
        new_data = W @ data  # (num_target, T)

        # --- Construir nuevo Raw con los canales objetivo ---
        info = mne.create_info(
            ch_names=[ch for ch in self.target_channels],
            sfreq=raw.info["sfreq"],
            ch_types="eeg",
        )
        raw_out = mne.io.RawArray(new_data, info, verbose=False)
        raw_out.set_annotations(raw.annotations)

        return raw_out

    def __repr__(self) -> str:
        return f"SpatialInterpolator(n_target={len(self.target_channels)})"
