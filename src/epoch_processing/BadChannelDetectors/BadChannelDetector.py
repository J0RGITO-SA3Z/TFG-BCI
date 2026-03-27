'''
Clase base para detectar canales malos en datos de EEG. Esta clase define la estructura general para los detectores de canales malos,
pero no implementa la lógica específica de detección, que debe ser proporcionada por las subclases.

Ejemplo de uso:
    from epoch_processing.BadChannelDetectors import BadChannelDetector

    class VarianceBadChannelDetector(BadChannelDetector):
        def is_bad_channel(self, X: np.ndarray) -> bool:
            return np.var(X) < self.threshold

    detector = VarianceBadChannelDetector(threshold=0.5)
    bad_channels = detector.process(eeg_data)
'''

import numpy as np
from abc import ABC, abstractmethod

class BadChannelDetector(ABC):
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, X: np.ndarray) -> list[int]:
        '''
        Procesa cada canal y devuelve una lista de índices de canales considerados malos.
        recive un array de forma (n_channels, n_samples)
        '''
        bad_channels = []
        for channel in range(X.shape[0]):
            if self.is_bad_channel(X[channel]):
                bad_channels.append(channel)
        return bad_channels

    @abstractmethod
    def is_bad_channel(self, X: np.ndarray) -> bool:
        # Implement logic to determine if the channel is bad based on the threshold
        pass