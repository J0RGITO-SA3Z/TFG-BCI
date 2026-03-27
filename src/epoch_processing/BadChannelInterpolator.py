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

from typing import List, Optional
from collections.abc import Iterable

from epoch_processing import EpochProcessor
from epoch_processing.BadChannelDetectors.BadChannelDetector import BadChannelDetector

# Channel spatial interpolation
from epoch_processing.SpatialInterpolator import SpatialInterpolator

class BadChannelInterpolator(EpochProcessor):
    
    def __init__(self, channels_max = None, detectors: Iterable[BadChannelDetector] | None = None,actual_channel_positions: Optional[List[str]] = None):
        self.detectors: list[BadChannelDetector] = list(detectors) if detectors else []
        self.bad_channel_list = []
        self.actual_channel_positions = actual_channel_positions
        self.channels_max = channels_max

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
        3. Elimina temporalmente esos canales del trial
        4. Reconstruye el trial completo mediante interpolacion espacial

        La forma de salida siempre se conserva como `(B, C, T)`.
        Las etiquetas `y` no se modifican.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        batch_size, n_channels, _ = X.shape
        processed_trials = []
        y_labels = []

        for batch_idx in range(batch_size):
            trial_batch = X[batch_idx:batch_idx + 1]
            bad_indices = set()

            for detector in self.detectors:
                detected = detector.process(trial_batch)
                bad_indices.update(self._normalize_detector_output(detected, n_channels))
            if(self.channels_max is not None and len(bad_indices) <= self.channels_max):      
                processed_trial = self._interpolate_trial(
                    trial=X[batch_idx],
                    bad_indices=sorted(bad_indices),
                )
                processed_trials.append(processed_trial)
                y_labels.append(y[batch_idx] if y is not None else None)

        X_processed = np.stack(processed_trials, axis=0)
        Y_processed = np.array(y_labels)
        return X_processed, Y_processed
