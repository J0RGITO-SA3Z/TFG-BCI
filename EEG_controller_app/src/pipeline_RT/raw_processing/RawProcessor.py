"""
Interfaz abstracta para procesadores de señales EEG sobre objetos MNE Raw.
"""

from abc import ABC, abstractmethod
import mne

class RawProcessor(ABC):
    """
    Clase base abstracta para cualquier procesador de señales EEG.

    Cada subclase debe implementar el método `process`, que recibe un
    ``mne.io.Raw`` y devuelve otro ``mne.io.Raw`` procesado.
    """

    @abstractmethod
    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Aplica el procesamiento sobre una copia del Raw recibido.

        Args:
            raw: Objeto ``mne.io.Raw`` de entrada.

        Returns:
            Nuevo ``mne.io.Raw`` con el procesamiento aplicado.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
