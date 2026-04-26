import numpy as np
from .BadChannelDetector import BadChannelDetector

class AmplitudeThresholdDetector(BadChannelDetector):
    """
    Detecta canales malos por pico de amplitud.

    Un canal es malo si la diferencia pico a pico (max - min) supera ``threshold``.
    El umbral típico en MI-BCI es 100 µV.

    Parameters
    ----------
    threshold : float
        Amplitud máxima permitida en µV (por defecto 100).
    """

    def __init__(self, threshold: float = 100.0) -> None:
        super().__init__(threshold)

    def is_bad_channel(self, X: np.ndarray) -> bool:
        if abs(float(np.max(X)) - float(np.min(X))) > self.threshold:
            print(f"    Canal detectado como RUIDOSO (pico a pico={float(np.max(X)) - float(np.min(X)):.2f} µV > {self.threshold} µV)")
        return bool(abs(float(np.max(X)) - float(np.min(X))) > self.threshold)
