"""
Contenedor que mantiene una lista ordenada de :class:`EpochProcessor` y los
aplica secuencialmente sobre un ``mne.Epochs``.

Ejemplo de uso::

    from epoch_processing import (
        EpochProcessorPipeline, SpatialInterpolator,
        EuclideanAlignment, EpochNormalizer,
    )

    pipeline = EpochProcessorPipeline([
        SpatialInterpolator(target_channels, channel_positions),
        EuclideanAlignment(),
        EpochNormalizer(),
    ])

    epochs_out = pipeline.process(epochs_in)
"""

from __future__ import annotations
from typing import Iterable
import mne

from epoch_processing.EpochProcessor import EpochProcessor

class EpochProcessorPipeline:

    def __init__(self, processors: Iterable[EpochProcessor] | None = None) -> None:
        self._processors: list[EpochProcessor] = list(processors) if processors else []

    def add(self, processor: EpochProcessor) -> "EpochProcessorPipeline":
        """Agrega un nuevo procesador al final de la cadena."""
        if not isinstance(processor, EpochProcessor):
            raise TypeError(
                f"Se esperaba una instancia de EpochProcessor, "
                f"se recibió {type(processor).__name__}"
            )
        self._processors.append(processor)
        return self

    def clear(self) -> None:
        """Elimina todos los procesadores de la cadena."""
        self._processors.clear()

    @property
    def processors(self) -> list[EpochProcessor]:
        """Devuelve una copia de la lista de procesadores."""
        return list(self._processors)

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Aplica todos los procesadores en orden sobre los Epochs de entrada.
        """
        for proc in self._processors:
            epochs = proc.process(epochs)
        return epochs

    def __repr__(self) -> str:
        steps = ", ".join(repr(p) for p in self._processors)
        return f"EpochProcessorPipeline([{steps}])"

    def __len__(self) -> int:
        return len(self._processors)
