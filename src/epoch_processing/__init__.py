"""
Paquete ``epoch_processing`` — procesadores modulares para señales EEG (MNE Epochs).

Uso rápido::

    from epoch_processing import (
        EpochProcessorPipeline, SpatialInterpolator,
        EuclideanAlignment, EpochNormalizer,
    )

    pipeline = EpochProcessorPipeline([
        SpatialInterpolator(target_channels, channel_positions),
        EuclideanAlignment(),
        EpochNormalizer(),
    ])
    epochs_processed = pipeline.process(epochs)
"""

from .EpochProcessor import EpochProcessor
from .SpatialInterpolator import SpatialInterpolator
from .EuclideanAlignment import EuclideanAlignment
from .EpochNormalizer import EpochNormalizer
from .EpochProcessorPipeline import EpochProcessorPipeline

__all__ = [
    "EpochProcessor",
    "SpatialInterpolator",
    "EuclideanAlignment",
    "EpochNormalizer",
    "EpochProcessorPipeline",
]
