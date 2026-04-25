from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Optional, Type

from EEG_app.src.components.RT_pipe_components.MIGames.MIGame import MIGame


class MIGameProcess:
    """Lanza cualquier MIGame en un proceso separado y gestiona su ciclo de vida."""

    def __init__(self, game_cls: Type[MIGame]) -> None:
        self.game_cls = game_cls
        self.stop_event = mp.Event()
        self.process: Optional[mp.Process] = None
        self._recv_pipe: Optional[Connection] = None
        self._send_pipe: Optional[Connection] = None

    @staticmethod
    def _process_target(
        recv_pipe: Connection,
        stop_event: mp.synchronize.Event,
        filename: Optional[str],
        game_cls: Type[MIGame],
    ) -> None:
        game = game_cls(recv_pipe, stop_event)
        game.start(filename)

    def start(self, filename: Optional[str] = None) -> None:
        if self.process is not None and self.process.is_alive():
            return

        self.stop_event.clear()
        self._recv_pipe, self._send_pipe = mp.Pipe(duplex=False)

        self.process = mp.Process(
            target=self._process_target,
            args=(self._recv_pipe, self.stop_event, filename, self.game_cls),
            daemon=True,
        )
        self.process.start()

    def send_msg(self, decider, prediction, prob) -> None:
        if self.process is None or not self.process.is_alive() or self._send_pipe is None:
            raise RuntimeError("MIGameProcess no está iniciado")
        self._send_pipe.send((decider, prediction, prob))

    def stop(self) -> None:
        self.stop_event.set()

        if self.process is not None:
            self.process.join()
            self.process = None

        if self._send_pipe is not None:
            self._send_pipe.close()
            self._send_pipe = None

        if self._recv_pipe is not None:
            self._recv_pipe.close()
            self._recv_pipe = None
