from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional, Tuple, Any


class MIGame(ABC):
    """Interfaz base para juegos de MI. Se encarga de recibir predicciones del interprete y acumularlas para tomar decisiones cada ciclo."""

    def __init__(self, pipe: Connection, stop_event: Event) -> None:
        self.pipe = pipe
        self.stop_event = stop_event

    def read_msg(self) -> Tuple[Any, Any, Any]:
        """Lee un mensaje del pipe y lo devuelve como (Decider, Prediction, Prob)."""
        msg = self.pipe.recv()

        if not isinstance(msg, tuple) or len(msg) != 3:
            raise ValueError("Formato de mensaje invalido. Se esperaba (Decider, Prediction, Prob)")
        decider, prediction, prob = msg
        return decider, prediction, prob

    @abstractmethod
    def start(self, filename: Optional[str] = None) -> None:
        """Inicia la logica del juego. Debe ejecutarse en un proceso separado y escuchar el pipe hasta que se le indique que se detenga."""
        raise NotImplementedError()
