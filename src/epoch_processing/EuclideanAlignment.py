"""
Euclidean Alignment sobre Epochs.
"""
import os, sys
import numpy as np
import mne

from epoch_processing.EpochProcessor import EpochProcessor

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