import numpy as np
from BadChannelDetector import BadChannelDetector

class VarianceDetector(BadChannelDetector):
    """
    Detecta canales malos por varianza anómala.

    Un canal es malo si:
      - su varianza supera ``threshold``  → canal ruidoso / artefactado, o
      - su varianza es prácticamente 0    → canal muerto / desconectado
        (se considera «muerto» si var < ``dead_threshold``).

    Parameters
    ----------
    threshold : float
        Varianza máxima permitida (µV²).
    dead_threshold : float
        Varianza mínima por debajo de la cual el canal se considera muerto.
        Por defecto 1e-10 (tolerancia numérica).
    """

    def __init__(self, threshold: float, dead_threshold: float = 1e-10) -> None:
        super().__init__(threshold)
        self.dead_threshold = dead_threshold

    def is_bad_channel(self, X: np.ndarray) -> bool:
        var = float(np.var(X))
        is_dead  = var < self.dead_threshold
        is_noisy = var > self.threshold
        return is_dead or is_noisy
