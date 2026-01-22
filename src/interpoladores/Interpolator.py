from abc import ABC, abstractmethod
import mne

class Interpolator(ABC):
    """
    Interpolador abstracto para datos EEG.
    """

    def interpolate(self, data):
        """
        Método público genérico.
        Decide qué interpolación aplicar según el tipo de entrada.
        """
        if isinstance(data, mne.Epochs):
            return self.interpolate_epochs(data)

        elif isinstance(data, mne.io.BaseRaw):
            return self.interpolate_raw(data)

        else:
            raise TypeError(
                f"Tipo no soportado para interpolación: {type(data)}"
            )

    @abstractmethod
    def interpolate_epochs(self, epochs: mne.Epochs):
        """
        Interpola datos a nivel de Epochs.
        """
        pass

    @abstractmethod
    def interpolate_raw(self, raw: mne.io.BaseRaw):
        """
        Interpola datos a nivel de Raw.
        """
        pass