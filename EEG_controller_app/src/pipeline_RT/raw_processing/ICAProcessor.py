"""
Procesador: eliminación de artefactos mediante ICA sobre MNE Raw.
"""

import mne
from raw_processing.RawProcessor import RawProcessor

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
        n_components: int = 0.999999,
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

        #Detección automática de componentes relacionados con EOG, EMG y ECG
        
        muscle_indices, _ = ica.find_bads_muscle(raw, verbose=False)
        #ecg_indices, _ = ica.find_bads_ecg(raw, verbose=False)
        #eog_indices, _ = ica.find_bads_eog(raw, verbose=False) # EOG = parpadeo y movimientos oculares pero no se puede eliminar porque para ello hay que teber un electrodo específico
        ica.exclude = list(set( muscle_indices))

        raw = ica.apply(raw, verbose=False)
        return raw

    def __repr__(self) -> str:
        return (
            f"ICAProcessor(n_components={self.n_components}, "
            f"method='{self.method}', random_state={self.random_state})"
        )
