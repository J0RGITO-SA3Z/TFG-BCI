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

    def __init__(
        self,
        channels_max=None,
        detectors: Iterable[BadChannelDetector] | None = None,
        actual_channel_positions: Optional[List[str]] = None,
        print_history: bool = False,
    ):
        self.detectors: list[BadChannelDetector] = list(detectors) if detectors else []
        self.bad_channel_list: list[list[str]] = []
        self.discarded_epoch_indices: list[int] = []
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

    def _run_detection(self, X: np.ndarray) -> tuple[list[list[str]], list[int]]:
        """Corre los detectores sobre X (B,C,T) y devuelve (bad_channel_list, discarded_indices)."""
        batch_size, n_channels, _ = X.shape
        bad_channel_list: list[list[str]] = []
        discarded_indices: list[int] = []

        for batch_idx in range(batch_size):
            trial = X[batch_idx]
            bad_indices: set[int] = set()

            for detector in self.detectors:
                detected = detector.process(trial)
                bad_indices.update(self._normalize_detector_output(detected, n_channels))

            bad_names = (
                [self.actual_channel_positions[i] for i in bad_indices]
                if self.actual_channel_positions else []
            )
            bad_channel_list.append(bad_names)

            if self.channels_max is not None and len(bad_indices) > self.channels_max:
                discarded_indices.append(batch_idx)

        return bad_channel_list, discarded_indices

    def _bad_indices_for_epoch(self, batch_idx: int) -> list[int]:
        """Devuelve los índices de canales malos almacenados para un epoch dado."""
        if batch_idx >= len(self.bad_channel_list):
            return []
        bad_names = self.bad_channel_list[batch_idx]
        if not bad_names or self.actual_channel_positions is None:
            return []
        return sorted(
            self.actual_channel_positions.index(name)
            for name in bad_names
            if name in self.actual_channel_positions
        )

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def detect_only(self, X: np.ndarray) -> tuple[list[list[str]], list[int]]:
        """Solo detecta canales malos y epochs a descartar; almacena los resultados.

        Llama a este método sobre los datos del pipeline de detección y luego
        usa `apply_detected` sobre los datos del pipeline final.

        Returns
        -------
        bad_channel_list : list[list[str]]
            Canales malos por epoch (lista de nombres).
        discarded_epoch_indices : list[int]
            Índices de epochs que superan `channels_max`.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        bad_channel_list, discarded_indices = self._run_detection(X)
        self.bad_channel_list = bad_channel_list
        self.discarded_epoch_indices = discarded_indices

        if self.print_history_enabled:
            self._print_history(bad_channel_list)

        return bad_channel_list, discarded_indices

    def apply_detected(
        self, X: np.ndarray, y: np.ndarray | None = None, interpolate: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Aplica los resultados de `detect_only` a datos nuevos.

        Descarta los epochs listados en `discarded_epoch_indices`.
        Si `interpolate=True`, además reconstruye los canales malos almacenados
        en `bad_channel_list`. Si `interpolate=False`, los epochs se conservan
        tal cual sin tocar los canales malos.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        batch_size, n_channels, n_times = X.shape
        discarded_set = set(self.discarded_epoch_indices)
        processed_trials: list[np.ndarray] = []
        y_labels: list = []
        discarded_trials: list[np.ndarray] = []

        for batch_idx in range(batch_size):
            trial = X[batch_idx]

            if batch_idx in discarded_set:
                discarded_trials.append(trial)
                continue

            if interpolate:
                bad_indices = self._bad_indices_for_epoch(batch_idx)
                processed_trials.append(self._interpolate_trial(trial, bad_indices))
            else:
                processed_trials.append(trial.copy())
            y_labels.append(y[batch_idx] if y is not None else None)

        X_processed = (
            np.stack(processed_trials, axis=0)
            if processed_trials
            else np.empty((0, n_channels, n_times))
        )
        Y_processed = np.array(y_labels)

        return X_processed, Y_processed

    def process_np(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detecta e interpola en un único paso (comportamiento original).

        Procesa `X` con forma `(B, C, T)`:
        1. Ejecuta todos los detectores de canales malos.
        2. Si `channels_max` está definido y se supera, descarta el epoch.
        3. En caso contrario, interpola los canales malos.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,C,T), got shape {X.shape}")

        bad_channel_list, discarded_indices = self._run_detection(X)
        self.bad_channel_list = bad_channel_list
        self.discarded_epoch_indices = discarded_indices

        batch_size, n_channels, n_times = X.shape
        discarded_set = set(discarded_indices)
        processed_trials: list[np.ndarray] = []
        y_labels: list = []
        discarded_trials: list[np.ndarray] = []

        for batch_idx in range(batch_size):
            trial = X[batch_idx]

            if batch_idx in discarded_set:
                discarded_trials.append(trial)
                continue

            bad_indices = self._bad_indices_for_epoch(batch_idx)
            processed_trials.append(self._interpolate_trial(trial, bad_indices))
            y_labels.append(y[batch_idx] if y is not None else None)

        X_processed = (
            np.stack(processed_trials, axis=0)
            if processed_trials
            else np.empty((0, n_channels, n_times))
        )
        Y_processed = np.array(y_labels)

        if self.print_history_enabled:
            self._print_history(bad_channel_list)

        return X_processed, Y_processed

    def process(self, epochs: mne.Epochs) -> mne.Epochs:
        """Implementación de la interfaz `EpochProcessor.process`."""
        X = epochs.get_data()
        X_processed, _ = self.process_np(X, None)

        new_channels = self.actual_channel_positions if self.actual_channel_positions is not None else None
        return self._to_epochs(X_processed, epochs, new_channels=new_channels)

    # ── Helpers de visualización ───────────────────────────────────────────────

    def _print_history(self, history: list[list[str]] | None = None) -> None:
        """Imprime en formato tabular un resumen por epoch con los canales malos."""
        if history is None:
            history = self.bad_channel_list

        if not history:
            print("No hay historial de canales malos para mostrar.")
            return

        idx_width = max(len(str(len(history) - 1)), len("Epoch"))
        channels_strs = [", ".join(ch) if ch else "-" for ch in history]
        chan_width = max(max((len(s) for s in channels_strs), default=0), len("Bad channels"))

        header = f"{'Epoch'.ljust(idx_width)} | {'Bad channels'.ljust(chan_width)}"
        sep = f"{'-' * idx_width}-+-{'-' * chan_width}"
        print(header)
        print(sep)

        for i, ch_str in enumerate(channels_strs):
            print(f"{str(i).ljust(idx_width)} | {ch_str.ljust(chan_width)}")

