import mne
import numpy as np
from Interpolator import Interpolator

class ConstantInterpolator(Interpolator):
    """
    Interpolador que reemplaza completamente los canales malos
    por un valor constante. (lo normal serria usar 0)
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def interpolate_epochs(self, epochs: mne.Epochs):
        epochs = epochs.copy()

        bads = epochs.info['bads']
        if not bads:
            return epochs

        ch_names = epochs.info['ch_names']
        bad_idx = [ch_names.index(ch) for ch in bads if ch in ch_names]

        data = epochs._data  # (n_epochs, n_channels, n_times)
        data[:, bad_idx, :] = self.value

        return epochs

    def interpolate_raw(self, raw: mne.io.BaseRaw):
        raw = raw.copy()

        bads = raw.info['bads']
        if not bads:
            return raw

        ch_names = raw.info['ch_names']
        bad_idx = [ch_names.index(ch) for ch in bads if ch in ch_names]

        data = raw._data  # (n_channels, n_times)
        data[bad_idx, :] = self.value

        return raw