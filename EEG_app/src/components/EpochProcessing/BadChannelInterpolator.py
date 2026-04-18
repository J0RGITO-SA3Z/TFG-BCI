'''
Este módulo define la clase `BadChannelInterpolator`, que es un `EpochProcessor` diseñado para detectar y corregir canales malos en datos EEG mediante interpolación espacial.
La clase utiliza una lista de detectores de canales malos (instancias de `BadChannelDetector`) para identificar qué canales deben ser corregidos, y luego aplica interpolación espacial para reconstruir los datos de esos canales.
Ejemplo de uso:
    from epoch_processing import BadChannelInterpolator
    from epoch_processing.BadChannelDetectors import VarianceBadChannelDetector

    # Crear un detector de canales malos basado en la varianza
    variance_detector = VarianceBadChannelDetector(threshold=0.5)

    # Crear el interpolador de canales malos con el detector y las posiciones de los canales
    interpolator = BadChannelInterpolator(
        detectors=[variance_detector],
        actual_channel_positions=...
    )

    # Procesar los datos EEG (X) con el interpolador
    X_processed, y_processed = interpolator.process_np(X, y)
'''
import numpy as np
import mne

from typing import List, Optional
from collections.abc import Iterable

from components.EpochProcessing.EpochProcessor import EpochProcessor
from components.EpochProcessing.BadChannelDetectors.BadChannelDetector import BadChannelDetector

# Channel spatial interpolation
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator

class BadChannelInterpolator(EpochProcessor):
    
    def __init__(self, channels_max = None, detectors: Iterable[BadChannelDetector] | None = None,actual_channel_positions: Optional[List[str]] = None, print_history: bool = False):
        self.detectors: list[BadChannelDetector] = list(detectors) if detectors else []
        self.bad_channel_list = []
        self.actual_channel_positions = actual_channel_positions
        self.channels_max = channels_max
        self.print_history_enabled = bool(print_history)

    def process_epoch(self, epoch):
        return epoch

    def _normalize_detector_output(self, detector_output, n_channels: int) -> list[int]:
        if detector_output is None:
            return []

        if isinstance(detector_output, Iterable) and not isinstance(detector_output, (str, bytes)):
            candidates = detector_output
        else:
            candidates = [detector_output]

        bad_indices = []
        for idx in candidates:
            try:
                channel_idx = int(idx)
            except (TypeError, ValueError):
                continue

            if 0 <= channel_idx < n_channels:
                bad_indices.append(channel_idx)

        return bad_indices

    def _interpolate_trial(self, trial: np.ndarray, bad_indices: list[int]) -> np.ndarray:
        n_channels = trial.shape[0]

        if not bad_indices:
            return trial.copy()

        if self.actual_channel_positions is None:
            raise ValueError(
                "BadChannelInterpolator necesita actual_channel_positions para "
                "reconstruir los canales malos mediante interpolacion espacial."
            )

        if len(self.actual_channel_positions) != n_channels:
            raise ValueError(
                f"Length of actual_channel_positions ({len(self.actual_channel_positions)}) "
                f"does not match C ({n_channels})"
            )

        remaining_names = [
            channel_name
            for idx, channel_name in enumerate(self.actual_channel_positions)
            if idx not in bad_indices
        ]
        good_trial = np.delete(trial, bad_indices, axis=0)

        interpolator = SpatialInterpolator(
            target_channels=self.actual_channel_positions,
            actual_channel_positions=remaining_names,
        )
        interpolated_trial, _ = interpolator.process_np(good_trial[np.newaxis, ...], None)

        return interpolated_trial[0]
    
    def process_np(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Procesa `X` con forma `(B, C, T)`, donde:
        - `B` es el numero de trials del batch
        - `C` es el numero de canales EEG
        - `T` es el numero de muestras temporales por trial

        Para cada trial:
        1. Ejecuta todos los detectores de canales malos
        2. Unifica los indices devueltos por todos ellos
        3. Si el numero de canales malos es menor o igual a `channels_max` (si se ha definido), procede a corregir el trial, sino lo descarta
        4. Reconstruye los nuevos datos 

        La forma de salida siempre se conserva como `(B, C, T)`.
        Las etiquetas `y` no se modifican.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        batch_size, n_channels, _ = X.shape
        processed_trials = []
        y_labels = []
        history = []
        for batch_idx in range(batch_size):
            trial = X[batch_idx]
            bad_indices = set()

            for detector in self.detectors:
                detected = detector.process(trial)
                bad_indices.update(self._normalize_detector_output(detected, n_channels))
            bad_channels_names = [self.actual_channel_positions[idx] for idx in bad_indices] if self.actual_channel_positions else None
            history.append(bad_channels_names if bad_channels_names else [])
            if(self.channels_max is not None and len(bad_indices) <= self.channels_max):
                processed_trial = self._interpolate_trial(
                    trial=X[batch_idx],
                    bad_indices=sorted(bad_indices),
                )
                processed_trials.append(processed_trial)
                y_labels.append(y[batch_idx] if y is not None else None)

        if not processed_trials:
            # No trials processed (e.g. todos descartados) -> devolver arrays vacíos
            X_processed = np.empty((0, n_channels, X.shape[2]))
        else:
            X_processed = np.stack(processed_trials, axis=0)
        Y_processed = np.array(y_labels)

        # Guardar el historial y, si está habilitado, imprimir el resumen
        self.bad_channel_list = history
        if self.print_history_enabled:
            self._print_history(history)

        return X_processed, Y_processed

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        """Implementación de la interfaz `EpochProcessor.process`.

        Convierte `epochs` a numpy, aplica `process_np` y reconstruye
        un `mne.Epochs` con los datos procesados.
        """
        X = epochs.get_data()
        X_processed, _ = self.process_np(X, None)

        # Reconstruir epochs. Si se proporcionaron nombres de canales, úsalos.
        new_channels = self.actual_channel_positions if self.actual_channel_positions is not None else None
        return self._to_epochs(X_processed, epochs, new_channels=new_channels)

    def _print_history(self, history: list[list[str]] | None = None) -> None:
        """Imprime en formato tabular un resumen por epoch mostrando
        el índice de epoch y los canales detectados como malos.

        Cada fila: <epoch_index> | <canal1, canal2, ...>
        Si no hay canales malos para una epoch, aparece '-'.
        """
        if history is None:
            history = self.bad_channel_list

        if not history:
            print("No hay historial de canales malos para mostrar.")
            return

        # Calcular ancho de columnas
        idx_width = max(len(str(len(history) - 1)), len("Epoch"))
        channels_strs = [", ".join(ch) if ch else "-" for ch in history]
        chan_width = max(max((len(s) for s in channels_strs), default=0), len("Bad channels"))

        # Cabecera
        header = f"{'Epoch'.ljust(idx_width)} | {'Bad channels'.ljust(chan_width)}"
        sep = f"{'-' * idx_width}-+-{'-' * chan_width}"
        print(header)
        print(sep)

        for i, ch_str in enumerate(channels_strs):
            print(f"{str(i).ljust(idx_width)} | {ch_str.ljust(chan_width)}")
