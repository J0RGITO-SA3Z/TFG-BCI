import mne
import numpy as np
import time


class EEGSimulator:
    """
    Simula un EEGRecorder a partir de un fichero .fif ya grabado.
    Emite chunks de datos en tiempo real desde el hilo principal,
    bloqueando hasta que se complete la grabación.
    """

    def __init__(self, fif_path: str, batch_size: int = 64, loop: bool = False):
        """
        Parameters
        ----------
        fif_path  : ruta al fichero .fif con la grabación.
        batch_size: muestras por chunk enviado (default 64 ~ 250 ms a 250 Hz).
        loop      : si True, al llegar al final vuelve al principio indefinidamente.
        """
        self._raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        self._batch_size = batch_size
        self._loop = loop

        self._callbacks = []

        # Mapeo nombre_canal -> índice de fila en el array de datos
        self._channel_indexes = {
            name: i for i, name in enumerate(self._raw.ch_names)
        }

    # ------------------------------------------------------------------
    # Interfaz pública (compatible con EEGRecorder)
    # ------------------------------------------------------------------

    def register_callback(self, func):
        self._callbacks.append(func)

    def unregister_callback(self, func):
        self._callbacks.remove(func)

    def get_info(self) -> mne.Info:
        return self._raw.info

    def get_sfreq(self) -> float:
        return float(self._raw.info["sfreq"])

    def get_ch_names_ordered(self) -> list[str]:
        return list(self._raw.ch_names)

    def get_ch_types_ordered(self) -> list[str]:
        return [
            mne.channel_type(self._raw.info, i)
            for i in range(len(self._raw.ch_names))
        ]

    def get_channel_indexes(self) -> dict:
        return self._channel_indexes.copy()

    def iniciarGrabacion(self):
        """Emite los chunks en tiempo real desde el hilo principal. Bloquea hasta terminar."""
        data = self._raw.get_data()   # (n_channels, n_samples)
        sfreq = self.get_sfreq()
        batch_duration = self._batch_size / sfreq
        n_samples = data.shape[1]

        start = 0
        next_tick = time.perf_counter()

        while True:
            end = start + self._batch_size

            if end > n_samples:
                if not self._loop:
                    # Envía las muestras restantes si las hay
                    if start < n_samples:
                        chunk = data[:, start:n_samples]
                        self._fire_callbacks(chunk, chunk.shape[1])
                    break
                # Envía lo que queda y reinicia
                chunk = data[:, start:n_samples]
                if chunk.shape[1] > 0:
                    self._fire_callbacks(chunk, chunk.shape[1])
                start = 0
                continue

            chunk = data[:, start:end]
            self._fire_callbacks(chunk, self._batch_size)
            start += self._batch_size

            next_tick += batch_duration
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _fire_callbacks(self, chunk: np.ndarray, chunk_size: int):
        for func in self._callbacks:
            func(chunk, chunk_size)
