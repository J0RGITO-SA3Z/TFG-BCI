"""
Euclidean Alignment sobre Epochs.
"""
import os, sys
import numpy as np
import mne

from components.EpochProcessing.EpochProcessor import EpochProcessor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from scipy.linalg import fractional_matrix_power

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class EuclideanAlignment(EpochProcessor):
    """
    Aplica Euclidean Alignment sobre todos los trials de un ``mne.Epochs``.
    """

    def __init__(self,matrix: np.ndarray = None):
        self.matrix = matrix
        super().__init__()

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        data = epochs.get_data()  # (B, C, T)

        if self.matrix is None:
            # Primera llamada: calcular y guardar la matriz de referencia
            cov = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
            for i in range(data.shape[0]):
                cov[i] = np.cov(data[i])
            self.matrix = np.mean(cov, axis=0)

        sqrtRefEA = fractional_matrix_power(self.matrix, -0.5)
        XEA = np.zeros(data.shape)
        for i in range(data.shape[0]):
            XEA[i] = np.dot(sqrtRefEA, data[i])
        aligned = XEA.astype(np.float32)

        return self._to_epochs(aligned, epochs)

    def process_np(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if self.matrix is None:
            # Primera llamada: calcular y guardar la matriz de referencia
            cov = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                cov[i] = np.cov(X[i])
            self.matrix = np.mean(cov, axis=0)

        sqrtRefEA = fractional_matrix_power(self.matrix, -0.5)
        XEA = np.zeros(X.shape)
        for i in range(X.shape[0]):
            XEA[i] = np.dot(sqrtRefEA, X[i])

        return XEA.astype(np.float32), y
    
# Cálculo de la matriz media de las matrizes de covarianza de cada trial seccion sacada de utils de MIRepNet
def Calculate_EA_Matrix(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    refEA : numpy array
        reference matrix for Euclidean Alignment of shape (num_channels, num_channels)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)

    return refEA