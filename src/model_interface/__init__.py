"""
Paquete ``model_interface`` — interfaces de modelos de clasificación EEG.

Uso rápido::

    from model_interface import MiRepNetInterface

    model = MiRepNetInterface(weight_path="path/to/weights.pth", device="cpu")

    # Predicción de una muestra  (C, T)
    pred, probs, logits = model.predict(eeg_sample)

    # Predicción de un lote      (B, C, T)
    preds, probs, logits = model.predict_batch(eeg_batch)
"""

from .ModelInterface import ModelInterface
from .MiRepNetInterface import MiRepNetInterface

__all__ = [
    "ModelInterface",
    "MiRepNetInterface",
]
