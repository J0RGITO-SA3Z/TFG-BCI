"""
Procesador: filtro paso-banda (bandpass) sobre MNE Raw.
"""

import mne
from components.RawProcessing.RawProcessor import RawProcessor

class BandpassFilter(RawProcessor):
    """
    Aplica un filtro paso-banda al objeto Raw.

    Args:
        l_freq: Frecuencia de corte inferior (Hz).
        h_freq: Frecuencia de corte superior (Hz).
    """

    def __init__(self, l_freq: float, h_freq: float) -> None:
        self.l_freq = l_freq
        self.h_freq = h_freq

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq,method='fir', verbose=False)
        return raw

    def __repr__(self) -> str:
        return f"BandpassFilter(l_freq={self.l_freq}, h_freq={self.h_freq})"
