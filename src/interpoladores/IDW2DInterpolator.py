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

    def interpolate_bad_epochs(self, epochs: mne.Epochs):
        if not epochs.info['bads']:
            return epochs

        target_channels_aux = epochs.ch_names
        actual_channels_aux = [name for name in epochs.ch_names if name not in epochs.info['bads']]
        actual_idx = [epochs.ch_names.index(ch) for ch in actual_channels_aux]
        target_idx = [epochs.ch_names.index(ch) for ch in target_channels_aux]

        target_channels = [ch.upper() for ch in target_channels_aux]
        actual_channels = [ch.upper() for ch in actual_channels_aux]

        x = epochs.get_data()[:, actual_idx, :]

        B, C, T = x.shape
        num_target = len(target_channels)

        existing_pos = np.array([channel_positions[ch] for ch in actual_channels])
        target_pos = np.array([channel_positions[ch] for ch in target_channels])

        W = np.zeros((num_target, C))
        for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
            if target_ch in actual_channels:
                src_idx = actual_channels.index(target_ch)
                W[i, src_idx] = 1.0
            else:
                dist = cdist([pos], existing_pos)[0]
                weights = 1 / (dist + 1e-6)  
                weights /= weights.sum()     
                W[i] = weights
        
        padded = np.zeros((B, num_target, T))

        for b in range(B):
            padded[b] = W @ x[b]

        epochs_new = mne.EpochsArray(
            padded,
            epochs.info.copy(),
            epochs.events.copy(),
            epochs.event_id,
            epochs.tmin,
            epochs.baseline
        )

        return epochs_new

    def interpolate_bad_raw(self, raw: mne.io.BaseRaw):
        if not raw.info['bads']:
            return raw
        
        target_channels_aux = raw.ch_names
        actual_channels_aux = [ch for ch in raw.ch_names if ch not in raw.info['bads']]

        actual_idx = [raw.ch_names.index(ch) for ch in actual_channels_aux]

        target_channels = [ch.upper() for ch in target_channels_aux]
        actual_channels = [ch.upper() for ch in actual_channels_aux]

        x = raw.get_data(picks=actual_idx)

        C, T = x.shape
        num_target = len(target_channels)

        existing_pos = np.array([channel_positions[ch] for ch in actual_channels])
        target_pos = np.array([channel_positions[ch] for ch in target_channels])

        W = np.zeros((num_target, C))
        for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
            if target_ch in actual_channels:
                src_idx = actual_channels.index(target_ch)
                W[i, src_idx] = 1.0
            else:
                dist = cdist([pos], existing_pos)[0]
                weights = 1.0 / (dist + 1e-6)
                weights /= weights.sum()
                W[i] = weights

        padded = W @ x

        raw_new = mne.io.RawArray(
            padded,
            raw.info.copy()
        )

        return raw_new