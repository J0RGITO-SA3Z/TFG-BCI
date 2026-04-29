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
from typing import Tuple, List, Dict

class ModelInterface(ABC):

    @abstractmethod
    def finetuning(
        self,
        trainingData: np.ndarray,
        trainingLabels: np.ndarray,
        epochs: int,
        valData: Optional[np.ndarray] = None,
        valLabels: Optional[np.ndarray] = None,
    ) -> list:
        """
        Entrenamiento o fine-tuning del modelo.
        """
        ...

    @abstractmethod
    def predict(self, data: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predice una muestra EEG (C, T).

        Returns:
            label: clase predicha (string)
            probs: diccionario {clase: probabilidad}
        """
        ...

    @abstractmethod
    def predict_batch(self, data: np.ndarray) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Predice un lote EEG (B, C, T).

        Returns:
            labels: lista de clases predichas
            probs: lista de diccionarios {clase: probabilidad}
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Guarda los pesos del modelo en *path*.
        """
        ...

    @abstractmethod
    def get_constructor_params(self, weight_path: str) -> dict:
        """
        Devuelve un dict con los argumentos necesarios para reconstruir
        este objeto con ``NombreClase(**params)``.

        Args:
            weight_path: ruta al fichero de pesos guardado con :meth:`save`.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"