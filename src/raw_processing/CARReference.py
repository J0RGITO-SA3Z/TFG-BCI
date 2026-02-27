"""
Procesador: Common Average Reference (CAR) sobre MNE Raw.
"""

import mne
from .Processor import RawProcessor


class CARReference(RawProcessor):
    """
    Aplica re-referencia Common Average Reference (CAR) al objeto Raw.
    """

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return raw

    def __repr__(self) -> str:
        return "CARReference()"
