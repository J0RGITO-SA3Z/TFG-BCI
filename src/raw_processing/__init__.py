"""
Paquete ``raw_processing`` — procesadores modulares para señales EEG (MNE Raw).

Uso rápido::

    from raw_processing import (
        ProcessorPipeline, BandpassFilter, NotchFilter,
        Resampler, CARReference, ICAProcessor,
    )

    pipeline = ProcessorPipeline([
        BandpassFilter(8.0, 30.0),
        Resampler(250),
        CARReference(),
    ])
    raw_processed = pipeline.process(raw)
"""

from .RawProcessor import RawProcessor
from .BandpassFilter import BandpassFilter
from .NotchFilter import NotchFilter
from .Resampler import Resampler
from .CARReference import CARReference
from .ICAProcessor import ICAProcessor
from .AnnotationRenamer import AnnotationRenamer
from .RawProcessorPipeline import RawProcessorPipeline

__all__ = [
    "RawProcessor",
    "BandpassFilter",
    "NotchFilter",
    "Resampler",
    "CARReference",
    "ICAProcessor",
    "SpatialInterpolator",
    "AnnotationRenamer",
    "RawProcessorPipeline",
]
