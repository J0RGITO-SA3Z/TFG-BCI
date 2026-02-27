"""
Procesador: filtro notch sobre MNE Raw.
"""

import mne
from raw_processing.RawProcessor import RawProcessor

class NotchFilter(RawProcessor):
    """
    Aplica un filtro notch para eliminar interferencia de red eléctrica.

    Args:
        freq: Frecuencia a eliminar (Hz), por ejemplo 50.0 o 60.0.
    """

    def __init__(self, freq: float) -> None:
        self.freq = freq

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.notch_filter(freqs=self.freq, verbose=False)
        return raw

    def __repr__(self) -> str:
        return f"NotchFilter(freq={self.freq})"
