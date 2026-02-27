"""
Interfaz abstracta para modelos de clasificación de señales EEG.

Cada subclase debe implementar:
    - ``pretraining``  : entrenamiento / fine-tuning con datos (B, C, T).
    - ``predict``      : predicción de una única muestra (C, T).
    - ``predict_batch``: predicción de un lote de muestras (B, C, T).
"""

from abc import ABC, abstractmethod
from typing import Optional


import numpy as np


class ModelInterface(ABC):
    @abstractmethod
    def finetuning(
        self,
        trainingData: np.ndarray,
        trainingLabels: np.ndarray,
        epochs: int,
        valData: Optional[np.ndarray] = None,
        valLabels: Optional[np.ndarray] = None,
    ) -> float:
        """
        Entrenamiento o fine-tuning del modelo.
        """
        ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predice la clase de una única muestra EEG (C, T).
        """
        ...

    @abstractmethod
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Predice las clases para un lote EEG (B, C, T).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
