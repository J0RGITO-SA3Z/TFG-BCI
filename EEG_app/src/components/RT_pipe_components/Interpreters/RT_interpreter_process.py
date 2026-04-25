from __future__ import annotations

import multiprocessing as mp
from typing import Optional, Type

from .RT_interpreter import RT_interpreter
from .RT_interpreter_saver import RT_interpreter_saver


class RT_interpreter_process:
    def __init__(self, interpreter_cls: Type[RT_interpreter] = RT_interpreter_saver, game_pipe=None):
        self.interpreter_cls = interpreter_cls
        self.game_pipe = game_pipe
        self.stop_event = mp.Event()
        self.process = None
        self._recv_pipe = None
        self._send_pipe = None

    @staticmethod
    def _process_target(recv_pipe, stop_event, filename: Optional[str], interpreter_cls: Type[RT_interpreter], game_pipe):
        interpreter = interpreter_cls(recv_pipe, stop_event, game_pipe)
        interpreter.start(filename)

    def start(self, filename: Optional[str] = None) -> None:
        if self.process is not None and self.process.is_alive():
            return

        self.stop_event.clear()
        self._recv_pipe, self._send_pipe = mp.Pipe(duplex=False)

        self.process = mp.Process(
            target=self._process_target,
            args=(self._recv_pipe, self.stop_event, filename, self.interpreter_cls, self.game_pipe),
            daemon=True,
        )
        self.process.start()

    def send_msg(self, prediction, last_sample, last_timestamp) -> None:
        if self.process is None or not self.process.is_alive() or self._send_pipe is None:
            raise RuntimeError("RT_interpreter_process no esta iniciado")

        self._send_pipe.send((prediction, last_sample, last_timestamp))

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
