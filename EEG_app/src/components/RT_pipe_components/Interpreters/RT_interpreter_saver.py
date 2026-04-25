from __future__ import annotations

import csv
import time
from typing import Optional

from .RT_interpreter import RT_interpreter


class RT_interpreter_saver(RT_interpreter):
    """Interprete que acumula predicciones y las guarda en CSV al detenerse."""

    def __init__(self, pipe, stop_event, game_pipe=None):
        super().__init__(pipe, stop_event, game_pipe)
        self.predictions = []
        self.samples = []
        self.delays = []

    def _listen(self, filename: Optional[str] = None) -> None:
        while not self.stop_event.is_set():
            if self.pipe.poll(0.01):
                prediction, last_sample, last_timestamp, probs = self.read_msg()

                now = time.perf_counter()
                delay = now - last_timestamp

                self.predictions.append(prediction)
                self.samples.append(last_sample)
                self.delays.append(delay)

        if filename is not None:
            self._save(filename)

    def start(self, filename: Optional[str] = None) -> None:
        self._listen(filename)

    def _save(self, filename: str = "predictions_log.csv") -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction", "sample", "delay"])

            for p, s, d in zip(self.predictions, self.samples, self.delays):
                writer.writerow([p, s, d])
