"""
Clase base abstracta para proveedores de datos EEG.

Todos los proveedores deben implementar ``get_data()``, que devuelve
``(X, y, classes)`` con la misma semántica que ``load_moabb_data``:

- **X** : ``np.ndarray`` de forma ``(n_epochs, n_channels, n_samples)``
- **y** : ``np.ndarray`` de ``int64`` con las etiquetas codificadas
- **classes** : lista ordenada con los nombres originales de las clases
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class DataProvider(ABC):

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Devuelve los datos del dataset configurado.

        Returns:
            X       : np.ndarray (n_epochs, n_channels, n_samples)
            y       : np.ndarray int64 con etiquetas codificadas (0, 1, …)
            classes : lista ordenada de nombres de clase originales
        """
        ...

    @abstractmethod
    def get_channel_names(self) -> List[str]:
        """
        Devuelve la lista de nombres de canales del dataset.

        Returns:
            Lista de strings con los nombres de los canales EEG.
        """
        ...

