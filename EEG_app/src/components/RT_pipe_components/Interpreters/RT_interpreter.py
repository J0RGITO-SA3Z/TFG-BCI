from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional, Tuple, Any


class RT_interpreter(ABC):
    """Interfaz base para interpretes de tiempo real usados por RT_interpreter_process."""

    def __init__(self, pipe: Connection, stop_event: Event, game_pipe: Optional[Connection] = None) -> None:
        self.pipe = pipe
        self.stop_event = stop_event
        self.game_pipe = game_pipe

    def _send_to_game(self, prediction, info) -> None:
        if self.game_pipe is not None:
            self.game_pipe.send((prediction, info))

    def read_msg(self) -> Tuple[Any, Any, Any, Any]:
        """Lee un mensaje del pipe y lo devuelve como (prediction, last_sample, last_timestamp)."""
        msg = self.pipe.recv()

        if not isinstance(msg, tuple) or len(msg) != 4:
            raise ValueError("Formato de mensaje invalido. Se esperaba (prediction, last_sample, last_timestamp, probs)")

        prediction, last_sample, last_timestamp, probs = msg
        return prediction, last_sample, last_timestamp, probs
    
    @abstractmethod
    def start(self, filename: Optional[str] = None) -> None:
        """Inicia la logica del interprete y debe finalizar al activarse stop_event."""
        raise NotImplementedError()
