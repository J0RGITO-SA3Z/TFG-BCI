"""
Barajado aleatorio de epochs.

Reordena los trials de un ``mne.Epochs`` (o un array numpy) de forma
aleatoria, manteniendo la correspondencia entre datos y etiquetas.
"""

import numpy as np
import mne

from components.EpochProcessing.EpochProcessor import EpochProcessor


class EpochShuffler(EpochProcessor):
    """
    Baraja el orden de los epochs de forma aleatoria.

    Parameters
    ----------
    seed : int | None
        Semilla para reproducibilidad. ``None`` = aleatorio cada vez.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Interfaz MNE
    # ------------------------------------------------------------------

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        data = epochs.get_data()          # (B, C, T)
        events = epochs.events.copy()     # (B, 3)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(data))

        shuffled_data = data[idx]
        shuffled_events = events[idx]

        info = epochs.info.copy()
        return mne.EpochsArray(
            shuffled_data,
            info=info,
            events=shuffled_events,
            event_id=epochs.event_id,
            tmin=epochs.tmin,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Interfaz numpy
    # ------------------------------------------------------------------

    def process_np(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))

        X_shuffled = X[idx]
        y_shuffled = y[idx] if y is not None else None
        return X_shuffled, y_shuffled

    def __repr__(self) -> str:
        return f"EpochShuffler(seed={self.seed})"
