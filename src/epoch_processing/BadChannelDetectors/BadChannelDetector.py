import numpy as np
from abc import ABC, abstractmethod

class BadChannelDetector(ABC):
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, X: np.ndarray) -> list[str]:
        bad_channels = []
        for channel in epoch.channels:
            if self.is_bad_channel(channel):
                bad_channels.append(channel)
        return bad_channels

    @abstractmethod
    def is_bad_channel(self, X: np.ndarray) -> bool:
        # Implement logic to determine if the channel is bad based on the threshold
        pass