"""
Eliminación de epochs pertenecientes a una o varias clases.

Permite filtrar epochs descartando todos los trials cuya etiqueta
coincida con las clases indicadas, tanto sobre ``mne.Epochs`` como
sobre arrays numpy ``(X, y)``.
"""

import numpy as np
import mne

from epoch_processing.EpochProcessor import EpochProcessor


class ClassEventRemover(EpochProcessor):
    """
    Elimina todos los epochs cuya etiqueta pertenezca a las clases indicadas.

    Parameters
    ----------
    classes_to_remove : int | str | list[int | str]
        Clase(s) a eliminar. Puede ser un event_id numérico, un nombre
        de evento (str) o una lista con varios.
    """

    def __init__(self, classes_to_remove: int | str | list[int | str]) -> None:
        if isinstance(classes_to_remove, (int, str)):
            classes_to_remove = [classes_to_remove]
        self.classes_to_remove = classes_to_remove

    # ------------------------------------------------------------------
    # Interfaz MNE
    # ------------------------------------------------------------------

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        # Resolver nombres de eventos a ids numéricos
        ids_to_remove = set()
        for cls in self.classes_to_remove:
            if isinstance(cls, str):
                if cls in epochs.event_id:
                    ids_to_remove.add(epochs.event_id[cls])
            else:
                ids_to_remove.add(cls)

        # Máscara: conservar epochs cuyo event_id NO esté en ids_to_remove
        mask = ~np.isin(epochs.events[:, 2], list(ids_to_remove))

        data = epochs.get_data()[mask]        # (B', C, T)
        events = epochs.events[mask].copy()   # (B', 3)

        # Reconstruir event_id sin las clases eliminadas
        new_event_id = {
            name: eid for name, eid in epochs.event_id.items()
            if eid not in ids_to_remove
        }

        return mne.EpochsArray(
            data,
            info=epochs.info.copy(),
            events=events,
            event_id=new_event_id,
            tmin=epochs.tmin,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Interfaz numpy
    # ------------------------------------------------------------------

    def process_np(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if y is None:
            return X, y

        mask = ~np.isin(y, self.classes_to_remove)
        return X[mask], y[mask]

    def __repr__(self) -> str:
        return f"ClassEventRemover(classes_to_remove={self.classes_to_remove})"
