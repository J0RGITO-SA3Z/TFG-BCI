"""
Interfaz abstracta para procesadores de señales EEG sobre objetos MNE Epochs.
"""

from abc import ABC, abstractmethod
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

    @staticmethod
    def _to_epochs(data: np.ndarray, original: mne.Epochs) -> mne.Epochs:
        """Reconstruye ``mne.EpochsArray`` preservando info y eventos."""
        new_epochs = mne.EpochsArray(
            data,
            original.info.copy(),
            events=original.events,
            tmin=original.tmin,
            verbose=False,
        )
        return new_epochs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
