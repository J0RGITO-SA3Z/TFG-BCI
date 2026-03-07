"""
Euclidean Alignment no centrado en la media sobre Epochs.
"""
import os, sys
import numpy as np
import mne

from epoch_processing.EpochProcessor import EpochProcessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # sube desde src/epoch_processing -> src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrainedModels.MiRepNet.utils.utils import EA

class EuclideanAlignmentNotCentred(EpochProcessor):
    """
    Aplica Euclidean Alignment sobre todos los trials de un ``mne.Epochs``.
    """

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        data = epochs.get_data()  # (B, C, T)
        aligned =   self.EA_notcentred(data)
        return self._to_epochs(aligned, epochs)
    

    def EA_notcentred(self,X: np.ndarray) -> np.ndarray:
        """
        Euclidean Alignment aplicado correctamente sobre epochs (B, C, T).
        Calcula la covarianza media sobre todos los trials y la usa para blanquear
        cada trial individualmente — igual que hace MIRepNet en preentrenamiento.

        Args:
            X: (B, C, T)

        Returns:
            np.ndarray (B, C, T) alineado
        """
        B, C, T = X.shape
        print(X.shape)

        # Covarianza media entre todos los trials
        R_mean = np.mean([X[i] @ X[i].T / T for i in range(B)], axis=0)  # (C, C)
        eigvals, eigvecs = np.linalg.eigh(R_mean)
        eigvals = np.maximum(eigvals, 1e-10)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # R^{-1/2}
        return np.stack([whitening @ X[i] for i in range(B)], axis=0)  # (B, C, T)