"""
Procesador: eliminación de artefactos mediante ICA sobre MNE Raw.
"""

import mne
from .Processor import RawProcessor


class ICAProcessor(RawProcessor):
    """
    Aplica Independent Component Analysis (ICA) para eliminar artefactos
    oculares y musculares de la señal EEG.

    Args:
        n_components: Número de componentes ICA a estimar.
        method:       Algoritmo ICA (``"fastica"`` | ``"infomax"`` | ``"picard"``).
        random_state: Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_components: int = 15,
        method: str = "fastica",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.method = method
        self.random_state = random_state

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()

        ica = mne.preprocessing.ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            verbose=False,
        )
        ica.fit(raw, verbose=False)

        # Detección automática de artefactos oculares y musculares
        eog_indices: list[int] = []
        if "eog" in [ch.lower() for ch in raw.ch_names]:
            eog_indices, _ = ica.find_bads_eog(raw, verbose=False)

        muscle_indices, _ = ica.find_bads_muscle(raw, verbose=False)
        ica.exclude = list(set(eog_indices + muscle_indices))

        raw = ica.apply(raw, verbose=False)
        return raw

    def __repr__(self) -> str:
        return (
            f"ICAProcessor(n_components={self.n_components}, "
            f"method='{self.method}', random_state={self.random_state})"
        )
