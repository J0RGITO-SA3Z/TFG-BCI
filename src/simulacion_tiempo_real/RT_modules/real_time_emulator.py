import threading
import time
from typing import Optional, Sequence

import mne


class RealTimeEmulator:
    """
    Emula la llegada de datos EEG en tiempo real a partir de un archivo .fif.
    Replica la interfaz de callbacks de EEGRecorder para que los listeners
    reciban paquetes (chunk, chunk_size) como si vinieran del hardware.
    """

    def __init__(self, fif_path: str | Sequence[str] | None = None, chunk_samples: int = 10):
        """fif_path can be a path string, a sequence of paths, or None.
        If a sequence is provided, the raws will be concatenated.
        """
        if isinstance(fif_path, (list, tuple)):
            raws = [mne.io.read_raw_fif(p, preload=True, verbose=False) for p in fif_path]
            if not raws:
                raise ValueError("Empty list passed as fif_path")
            self.raw: mne.io.Raw = mne.concatenate_raws(raws)
        elif isinstance(fif_path, str):
            self.raw: mne.io.Raw = mne.io.read_raw_fif(fif_path, preload=True)
        else:
            raise ValueError("fif_path must be a path string or a sequence of path strings")
        self.sfreq: float = self.raw.info["sfreq"]
        self.chunk_samples = chunk_samples

        self._callbacks: list = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register_callback(self, func):
        self._callbacks.append(func)

    def unregister_callback(self, func):
        self._callbacks.remove(func)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def get_info(self) -> mne.Info:
        return self.raw.info

    def get_sfreq(self) -> float:
        return self.sfreq

    def get_ch_names(self) -> list[str]:
        return self.raw.ch_names

    def get_channel_indexes(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.raw.ch_names)}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait_until_finished(self, poll_interval: float = 0.1):
        while self.is_running():
            time.sleep(poll_interval)

    def _stream_loop(self):
        data = self.raw.get_data()
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

            elapsed = time.perf_counter() - t_start
            sleep_time = chunk_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
