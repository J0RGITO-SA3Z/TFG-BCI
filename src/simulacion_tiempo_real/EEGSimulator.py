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
        self._current_sample = 0  # posición persistente entre llamadas

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

    def iniciarGrabacion(self, segundos: float = None):
        """
        Emite chunks en tiempo real desde el hilo principal. Bloquea hasta terminar.

        Parameters
        ----------
        segundos : float | None
            Segundos máximos a retransmitir en esta llamada.
            Si es None, emite hasta el final del fichero (o indefinidamente si loop=True).
            Al volver a llamar, continúa desde donde se paró.
        """
        data = self._raw.get_data()   # (n_channels, n_samples)
        sfreq = self.get_sfreq()
        batch_duration = self._batch_size / sfreq
        n_samples = data.shape[1]

        hasta_el_final = segundos is None
        max_samples = int(segundos * sfreq) if not hasta_el_final else None
        emitidos = 0

        start = self._current_sample
        next_tick = time.perf_counter()

        while True:
            if not hasta_el_final and emitidos >= max_samples:
                break

            end = start + self._batch_size

            if end > n_samples:
                # Envía las muestras restantes
                if start < n_samples:
                    chunk = data[:, start:n_samples]
                    self._fire_callbacks(chunk, chunk.shape[1])
                    emitidos += chunk.shape[1]
                    start = n_samples

                if hasta_el_final or not self._loop:
                    break

                # Solo hace loop si se pidieron segundos concretos y loop=True
                start = 0
                continue

            chunk = data[:, start:end]
            self._fire_callbacks(chunk, self._batch_size)
            emitidos += self._batch_size
            start += self._batch_size

            next_tick += batch_duration
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._current_sample = start % n_samples  # guarda posición para la siguiente llamada

    def _fire_callbacks(self, chunk: np.ndarray, chunk_size: int):
        for func in self._callbacks:
            func(chunk, chunk_size)
