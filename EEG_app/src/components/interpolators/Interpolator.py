from abc import ABC, abstractmethod
import mne

class Interpolator(ABC):
    """
    Interpolador abstracto para datos EEG.
    """

    def interpolate_bads(self, data):
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
    def interpolate_bad_epochs(self, epochs: mne.Epochs):
        """
        Interpola datos marcados como bad a nivel de Epochs.
        """
        pass

    @abstractmethod
    def interpolate_bad_raw(self, raw: mne.io.BaseRaw):
        """
        Interpola datos marcados como bad a nivel de Raw.
        """
        pass

    @abstractmethod
    def interpolate_to_Raw(self, raw: mne.io.BaseRaw):
        """
        Interpola datos marcados como bad a nivel de Raw.
        """
        pass

    @abstractmethod
    def interpolate_to_epochs(self, raw: mne.io.BaseRaw):
        """
        Interpola datos marcados como bad a nivel de Raw.
        """
        pass