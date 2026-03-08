"""
Euclidean Alignment sobre Epochs.
"""
import os, sys
import numpy as np
import mne

from epoch_processing.EpochProcessor import EpochProcessor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # sube desde src/epoch_processing -> src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrainedModels.MiRepNet.utils.utils import EA

class EuclideanAlignment(EpochProcessor):
    """
    Aplica Euclidean Alignment sobre todos los trials de un ``mne.Epochs``.
    """

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        data = epochs.get_data()  # (B, C, T)
        aligned = EA(data).astype(np.float32)  
        return self._to_epochs(aligned, epochs)
    
    def process_np(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        aligned = EA(X).astype(np.float32)

        return aligned, y