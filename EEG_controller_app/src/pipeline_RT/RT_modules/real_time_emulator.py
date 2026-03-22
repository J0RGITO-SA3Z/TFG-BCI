import threading
import time
import numpy as np
import mne


class RealTimeEmulator:
    """
    Emula la llegada de datos EEG en tiempo real a partir de un archivo .fif.
    Replica la interfaz de callbacks de EEGRecorder para que los listeners
    reciban paquetes (chunk, chunk_size) como si vinieran del hardware.
    """

    def __init__(self, fif_path: str, chunk_samples: int = 10):
        """
        Parameters
        ----------
        fif_path : str
            Ruta al archivo .fif con un Raw de MNE.
        chunk_samples : int
            Número de muestras por paquete enviado a los listeners.
        """
        self.raw: mne.io.Raw = mne.io.read_raw_fif(fif_path, preload=True)
        self.sfreq: float = self.raw.info["sfreq"]
        self.chunk_samples = chunk_samples

        self._callbacks: list = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Listeners ──────────────────────────────────────────────────────

    def register_callback(self, func):
        """Registra una función que será llamada con (chunk, chunk_size)."""
        self._callbacks.append(func)

    def unregister_callback(self, func):
        """Elimina una función registrada."""
        self._callbacks.remove(func)

    # ── Control ────────────────────────────────────────────────────────

    def start(self):
        """Inicia el hilo que emite paquetes en tiempo real."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Detiene el hilo de emisión."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # ── Info helpers ───────────────────────────────────────────────────

    def get_info(self) -> mne.Info:
        return self.raw.info

    def get_sfreq(self) -> float:
        return self.sfreq

    def get_ch_names(self) -> list[str]:
        return self.raw.ch_names

    # ── Hilo interno ──────────────────────────────────────────────────

    def _stream_loop(self):
        data = self.raw.get_data()  # (n_channels, n_samples)
        n_samples = data.shape[1]
        chunk_duration = self.chunk_samples / self.sfreq

        idx = 0
        while idx < n_samples and not self._stop_event.is_set():
            t_start = time.perf_counter()

            end = min(idx + self.chunk_samples, n_samples)
            chunk = data[:, idx:end]
            chunk_size = chunk.shape[1]

            for func in self._callbacks:
                func(chunk, chunk_size)

            idx = end

            # Espera el tiempo restante para respetar la frecuencia de muestreo
            elapsed = time.perf_counter() - t_start
            sleep_time = chunk_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
