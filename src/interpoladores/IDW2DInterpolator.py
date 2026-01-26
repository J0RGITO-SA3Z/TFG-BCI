import mne
import numpy as np
from scipy.spatial.distance import cdist

from Interpolator import Interpolator
from channel_list import *

class IDW2DInterpolator(Interpolator):
    """
    Interpolador que podera los diferentes electrodos de manera
    inversamente proporcional a la distancia (IDW). Cuanto más lejos
    esté un electrodo malo, menos influirá en la interpolación.
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def interpolate_epochs(self, epochs: mne.Epochs):
        data = epochs.get_data()
        E, C, T = data.shape # Epochs, Channels, Times

        targetChannels = epochs.info['bads']
        if not targetChannels:
            return epochs
        
        targetIndexes = [epochs.ch_names.index(ch) for ch in targetChannels if ch in epochs.ch_names]
        
        actualChannels = [name for name in epochs.ch_names if name not in targetChannels]

        x_epochs = epochs.copy().pick_channels(actualChannels)
        x = epochs.copy().pick_channels(actualChannels).get_data()

        existing_pos = np.array([channel_positions[ch] for ch in actualChannels])
        targetChannel_pos = np.array([channel_positions[ch] for ch in targetChannels])

        W = np.zeros((len(targetChannels), len(actualChannels)))
        for i, (target_ch, pos) in enumerate(zip(targetChannels, targetChannel_pos)):
            if not target_ch in actualChannels:
                dist = cdist([pos],existing_pos)[0]
                weights = 1 / (dist + 1e-6)
                weights /= weights.sum()
                W[i]= weights

        res = epochs.copy()

        for i, ind in enumerate(targetIndexes):
            for e in range(E):
                for t in range(T):
                    res._data[e, ind, t] = np.sum(W[i] * x[e, :, t])

        return res

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