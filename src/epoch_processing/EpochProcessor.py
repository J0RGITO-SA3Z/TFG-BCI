"""
Interfaz abstracta para procesadores de señales EEG sobre objetos MNE Epochs.
"""

from abc import ABC, abstractmethod
import numpy as np
import mne

class EpochProcessor(ABC):
    """
    Clase base abstracta para cualquier procesador de epochs EEG.
    """

    @abstractmethod
    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Aplica el procesamiento sobre los epochs recibidos.
        """
        pass

    @abstractmethod
    def process_np(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Aplica el procesamiento sobre los epochs recibidos en formato np.
        """
        pass

    @staticmethod
    def _to_epochs(data: np.ndarray, original: mne.Epochs, new_channels = None) -> mne.Epochs:
        """Reconstruye ``mne.EpochsArray`` preservando info y eventos."""
        info = original.info.copy()
        if new_channels is not None:
            info = mne.create_info(
                ch_names=new_channels,
                sfreq=original.info['sfreq'],
                ch_types=['eeg'] * len(new_channels)
            )

        new_epochs = mne.EpochsArray(
            data,
            info=info,
            events=original.events,
            event_id=original.event_id,
            tmin=original.tmin,
            verbose=False,
        )
        return new_epochs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
