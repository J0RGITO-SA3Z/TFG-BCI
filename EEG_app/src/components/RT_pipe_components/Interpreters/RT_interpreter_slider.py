from __future__ import annotations

import csv
import time
from typing import Optional

from .RT_interpreter import RT_interpreter
from .Decision_filters import Mayoria, ScorePonderado, MediaProbabilidades, IntegradorFuga, ExponentialSmoothing

WINDOW_SIZE = 5.0 # en SAMPLES, no en segundos, porque queremos una ventana deslizante de 5 muestras

class RT_interpreter_slider(RT_interpreter):
    """Interprete que acumula predicciones y las guarda en CSV al detenerse."""

    def __init__(self, pipe, stop_event, game_pipe=None):
        super().__init__(pipe, stop_event, game_pipe)

        # resultados finales (una fila por ventana)
        self.final_predictions = []
        self.final_info = []
        self.final_samples = []
        self.final_raw_probs = []

        # buffer de ventana actual
        self.window_predictions = []
        self.window_samples = []
        self.window_delays = []
        self.window_probs = []
        self.window_seen_samples: set = set()

        # =========================
        # TODO: AQUÍ CAMBIAMOS LA ESTRATEGIA
        #self.filter = Mayoria()
        #self.filter = ScorePonderado()
        #self.filter = MediaProbabilidades()
        #self.filter = IntegradorFuga(leak_r=0.1, leak_l=0.1, threshold=1.0, reset_on_decision=True)
        self.filter = ExponentialSmoothing(alpha=0.85, threshold=0.75)
        # =========================

    def _listen(self, filename: Optional[str] = None) -> None:
        while not self.stop_event.is_set():
            if self.pipe.poll(0.01):
                prediction, last_sample, last_timestamp, probs = self.read_msg()

                now = time.perf_counter()
                delay = now - last_timestamp

                # añadimos la predicción a la ventana (descartamos duplicados por sample)
                if last_sample not in self.window_seen_samples:
                    self.window_seen_samples.add(last_sample)
                    self.window_predictions.append(prediction)
                    self.window_samples.append(last_sample)
                    self.window_delays.append(delay)
                    self.window_probs.append((float(probs["left_hand"]), float(probs["right_hand"])))

                    # cuando la ventana está llena, procesamos y deslizamos
                    if len(self.window_predictions) > WINDOW_SIZE:
                        self.window_seen_samples.remove(self.window_samples[0])  # actualizamos el set de muestras vistas en la ventana
                        self.window_samples = self.window_samples[1:]
                        self.window_predictions = self.window_predictions[1:]
                        self.window_delays = self.window_delays[1:]
                        self.window_probs = self.window_probs[1:]

                        self._process_window()

        # procesar última ventana si queda algo
        self._process_window()

        if filename is not None:
            self._save(filename)


    def _save(self, filename: str = "predictions_log.csv") -> None:
        # union ordenada de todos los keys de info (distintos según el filtro)
        info_keys: list[str] = []
        seen: set[str] = set()
        for info in self.final_info:
            for k in info:
                if k not in seen:
                    info_keys.append(k)
                    seen.add(k)

        fieldnames = ["sample", "prediction", "p_left_raw", "p_right_raw"] + info_keys

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for pred, sample, (p_left, p_right), info in zip(
                self.final_predictions, self.final_samples, self.final_raw_probs, self.final_info
            ):
                writer.writerow({
                    "sample": sample,
                    "prediction": pred,
                    "p_left_raw": p_left,
                    "p_right_raw": p_right,
                    **info,
                })


    def start(self, filename: Optional[str] = None) -> None:
        self._listen(filename)


    def _process_window(self):
        """Procesa una ventana deslizante"""

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
        self.final_raw_probs.append(self.window_probs[-1])

        print(f"Predicción: {pred_final}, Info: {info}, Sample: {self.window_samples[-1]:.3f} s")

        self._send_to_game(pred_final, info)
