"""
Procesador: resampleo (cambio de frecuencia de muestreo) sobre MNE Raw.
"""

import mne
from components.RawProcessing.RawProcessor import RawProcessor

class Resampler(RawProcessor):
    """
    Re-muestrea el objeto Raw a la frecuencia indicada.

    Args:
        sfreq: Frecuencia de muestreo objetivo (Hz).
    """

    def __init__(self, sfreq: float) -> None:
        self.sfreq = sfreq

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.resample(sfreq=self.sfreq, verbose=False)
        return raw

    def __repr__(self) -> str:
        return f"Resampler(sfreq={self.sfreq})"
