"""
Procesador: renombrado de anotaciones (eventos) en MNE Raw.

Permite mapear los nombres de las annotations del Raw de entrada
a otros nombres, por ejemplo para traducir las etiquetas de un
experimento al formato que espera un modelo.
"""

from __future__ import annotations

import mne
from raw_processing.RawProcessor import RawProcessor

class AnnotationRenamer(RawProcessor):
    """
    Renombra las annotations de un Raw según un diccionario de mapeo.

    Las annotations cuyo ``description`` no aparezca en el mapeo se
    mantienen tal cual (o se eliminan si ``drop_unmapped=True``).

    Args:
        mapping:        Diccionario ``{nombre_original: nombre_nuevo}``.
        drop_unmapped:  Si ``True``, elimina las annotations que no estén
                        en el mapeo. Por defecto ``False``.

    Ejemplo::

        AnnotationRenamer({
            "IZQUIERDA": "left_hand",
            "DERECHA":   "right_hand",
            "ABAJO":     "feet",
            "DESCANSO":  "nothing",
        })
    """

    def __init__(
        self,
        mapping: dict[str, str],
        drop_unmapped: bool = False,
    ) -> None:
        self.mapping = mapping
        self.drop_unmapped = drop_unmapped

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        old_annot = raw.annotations

        onsets, durations, descriptions = [], [], []
        for onset, duration, desc in zip(
            old_annot.onset, old_annot.duration, old_annot.description
        ):
            if desc in self.mapping:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append(self.mapping[desc])
            elif not self.drop_unmapped:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append(desc)

        new_annot = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=old_annot.orig_time,
        )
        raw.set_annotations(new_annot)
        return raw

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k!r}→{v!r}" for k, v in self.mapping.items())
        return f"AnnotationRenamer({{{pairs}}})"
