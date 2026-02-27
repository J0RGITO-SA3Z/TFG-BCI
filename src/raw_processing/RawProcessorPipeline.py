"""
Contenedor que mantiene una lista ordenada de :class:`Processor` y los
aplica secuencialmente sobre un ``mne.io.Raw``.

Ejemplo de uso::

    from raw_processing import (
        ProcessorPipeline, BandpassFilter, Resampler, CARReference,
    )

    pipeline = ProcessorPipeline()
    pipeline.add(BandpassFilter(8.0, 30.0))
    pipeline.add(Resampler(250))
    pipeline.add(CARReference())

    raw_out = pipeline.process(raw_in)

También se puede construir de una vez::

    pipeline = ProcessorPipeline([
        BandpassFilter(8.0, 30.0),
        Resampler(250),
        CARReference(),
    ])
"""

from __future__ import annotations

from typing import Iterable

import mne

from .Processor import RawProcessor

class RawProcessorPipeline:

    def __init__(self, processors: Iterable[RawProcessor] | None = None) -> None:
        self._processors: list[RawProcessor] = list(processors) if processors else []

    def add(self, processor: RawProcessor) -> "RawProcessorPipeline":
        """
        Añade un procesador al final de la cadena.

        Returns:
            self, para permitir encadenamiento:
            ``pipeline.add(A).add(B).add(C)``
        """
        if not isinstance(processor, RawProcessor):
            raise TypeError(
                f"Se esperaba una instancia de Processor, se recibió {type(processor).__name__}"
            )
        self._processors.append(processor)
        return self

    def clear(self) -> None:
        """Elimina todos los procesadores de la cadena."""
        self._processors.clear()

    @property
    def processors(self) -> list[RawProcessor]:
        """Devuelve una copia de la lista de procesadores."""
        return list(self._processors)

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Aplica todos los procesadores en orden sobre el Raw de entrada.

        Antes de ejecutar la cadena, descarta automáticamente todos los
        canales que no sean EEG (acelerómetros, giroscopios, misc, etc.).

        Args:
            raw: Objeto ``mne.io.Raw`` de entrada.

        Returns:
            Nuevo ``mne.io.Raw`` resultante de aplicar toda la cadena.
        """
        raw = raw.copy().pick("eeg")
        for proc in self._processors:
            raw = proc.process(raw)
        return raw

    def __repr__(self) -> str:
        steps = ", ".join(repr(p) for p in self._processors)
        return f"ProcessorPipeline([{steps}])"

    def __len__(self) -> int:
        return len(self._processors)
