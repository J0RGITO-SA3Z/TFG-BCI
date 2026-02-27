"""
Normalización z-score por epoch sobre Epochs.

Normaliza cada trial de forma independiente: :math:`X_{norm}^{(i)} =
(X^{(i)} - \\mu) / (\\sigma + \\epsilon)` donde media y desviación se
calculan sobre los ejes canal × tiempo de cada trial.
"""

import numpy as np
import mne

from .EpochProcessor import EpochProcessor

class EpochNormalizer(EpochProcessor):
    """
    Normalización z-score por epoch (canal × tiempo).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        data = epochs.get_data()  # (B, C, T)
        normalized = self._normalize(data)
        return self._to_epochs(normalized, epochs)

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score por epoch sobre los ejes (canal, tiempo)."""
        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True)
        return (X - mean) / (std + self.eps)

    def __repr__(self) -> str:
        return f"EpochNormalizer(eps={self.eps})"
