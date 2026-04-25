from __future__ import annotations

import csv
import time
from typing import Optional

from .RT_interpreter import RT_interpreter
from .Decision_filters import Mayoria, ScorePonderado, MediaProbabilidades, IntegradorFuga

TIEMPO_VENTANA = 1.0 # en segundos

class RT_interpreter_resampler(RT_interpreter):
    """Interprete que acumula predicciones y las guarda en CSV al detenerse."""

    def __init__(self, pipe, stop_event, game_pipe=None):
        super().__init__(pipe, stop_event, game_pipe)

        # resultados finales (una fila por segundo)
        self.final_predictions = []
        self.final_info = []
        self.final_samples = []

        # buffer de ventana actual
        self.window_predictions = []
        self.window_samples = []
        self.window_delays = []
        self.window_probs = []
        self.window_seen_samples: set = set()

        self.window_start_time = None

        # =========================
        # TODO: AQUÍ CAMBIAMOS LA ESTRATEGIA
        #self.filter = Mayoria()
        #self.filter = ScorePonderado()
        #self.filter = MediaProbabilidades()
        self.filter = IntegradorFuga(leak_r=0.1, leak_l=0.3, threshold=1.0, reset_on_decision=True)
        # =========================

    def _listen(self, filename: Optional[str] = None) -> None:
        while not self.stop_event.is_set():
            if self.pipe.poll(0.01):
                prediction, last_sample, last_timestamp, probs = self.read_msg()

                now = time.perf_counter()
                delay = now - last_timestamp

                # inicializamos la ventana si no estaba ya
                if self.window_start_time is None:
                    self.window_start_time = now

                # añadimos la predicción a la ventana (descartamos duplicados por sample)
                if last_sample not in self.window_seen_samples:
                    self.window_seen_samples.add(last_sample)
                    self.window_predictions.append(prediction)
                    self.window_samples.append(last_sample)
                    self.window_delays.append(delay)
                    self.window_probs.append((float(probs["left_hand"]), float(probs["right_hand"])))

                # comprobamos si ha pasado 1 segundo
                if now - self.window_start_time >= TIEMPO_VENTANA:
                    self._process_window()
                    self.window_start_time = now  # empezamos una nueva ventana

        # procesar última ventana si queda algo
        self._process_window()

        if filename is not None:
            self._save(filename)


    def _save(self, filename: str = "predictions_log.csv") -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction", "sample"])

            for p, s in zip(self.final_predictions, self.final_samples):
                writer.writerow([p, s])


    def start(self, filename: Optional[str] = None) -> None:
        self._listen(filename)


    def _process_window(self):
        """Procesa una ventana de TIEMPO_VENTANA segundo(s)"""

        if not self.window_predictions:
            return

        pred_final, info = self.filter.decider(
            window_probs=self.window_probs,
            window_predictions=self.window_predictions,
            window_delays=self.window_delays,
        )

        self.final_predictions.append(pred_final)
        self.final_info.append(info)
        self.final_samples.append(self.window_samples[-1])

        print(f"Predicción: {pred_final}, Info: {info}, Sample: {self.window_samples[-1]:.3f} s")

        self._send_to_game(pred_final, info)

        # reset ventana
        self.window_predictions.clear()
        self.window_samples.clear()
        self.window_delays.clear()
        self.window_probs.clear()
        self.window_seen_samples.clear()
