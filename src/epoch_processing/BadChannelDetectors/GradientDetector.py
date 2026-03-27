import numpy as np
from BadChannelDetector import BadChannelDetector

class GradientDetector(BadChannelDetector):
    """
    Detecta canales malos por gradiente excesivo entre muestras consecutivas.

    Calcula la diferencia absoluta entre cada par de muestras adyacentes
    (np.diff) y marca el canal como malo si el valor máximo supera ``threshold``.
    Detecta saltos bruscos que indican artefactos de movimiento o
    saturación del amplificador.

    Parameters
    ----------
    threshold : float
        Gradiente máximo permitido entre muestras consecutivas (µV / muestra).
    """

    def __init__(self, threshold: float) -> None:
        super().__init__(threshold)

    def is_bad_channel(self, X: np.ndarray) -> bool:
        '''
        Se usa np.diff para calcular el gradiente de forma menos suave
        que np.gradient, (my util para deetcción que cambios bruscos y artefactos) 
        entre muestras consecutivas, luego se toma el valor absoluto y se encuentra el máximo. Si este máximo supera el umbral,
        se considera que el canal es malo.
        '''
        max_gradient = float(np.max(np.abs(np.diff(X))))
        return max_gradient > self.threshold