from typing import List, Optional, Sequence

import mne
import numpy as np
import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.DataProvider.DataProvider import DataProvider
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

_BANDPASS = BandpassFilter(8.0, 30.0)
_RENAMER  = AnnotationRenamer(LABEL_MAP)


class FifVentanaDataProvider(DataProvider):
    """
    Data provider que simula predicción en tiempo real:

    Para cada evento de interés en los archivos FIF:
      1. Recorta una ventana de ``window_size`` segundos desde el onset.
      2. Aplica el filtro de banda sólo sobre esa ventana.
      3. Toma los últimos ``epoch_duration`` segundos como epoch.

    Esto imita el escenario en el que, en tiempo real, el sistema recibe
    N segundos de señal, filtra y predice sobre los últimos 4 s.
    """

    def __init__(
        self,
        fif_paths: str | Sequence[str] = [],
        annotations_names: List[str] = ["left_hand", "right_hand"],
        window_size: float = 10.0,
        epoch_duration: float = 4.0,
        l_freq: float = 8.0,
        h_freq: float = 30.0,
    ) -> None:
        if isinstance(fif_paths, str):
            fif_paths = [fif_paths]

        self._fif_paths        = list(fif_paths)
        self._annotations_names = annotations_names
        self._window_size      = window_size
        self._epoch_duration   = epoch_duration
        self._bandpass         = BandpassFilter(l_freq, h_freq)
        self._renamer          = AnnotationRenamer(LABEL_MAP)

    # ------------------------------------------------------------------ #
    #  DataProvider interface                                              #
    # ------------------------------------------------------------------ #

    def get_data(self):
        all_X: List[np.ndarray] = []
        all_Y: List[str]        = []

        for path in self._fif_paths:
            print(f"Cargando (ventana) {path} ...")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            raw = raw.pick("eeg")

            # Renombrar anotaciones (IZQUIERDA → left_hand, etc.)
            raw = self._renamer.process(raw)

            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Quedarse sólo con los event_id de las clases que nos interesan
            event_id_sel = {k: v for k, v in event_id.items()
                            if k in self._annotations_names}
            if not event_id_sel:
                print(f"  AVISO: ninguna anotación de interés en {path}")
                continue

            inv_event_id = {v: k for k, v in event_id_sel.items()}
            sfreq         = raw.info["sfreq"]
            epoch_samples = int(self._epoch_duration * sfreq)

            for sample_idx, _, event_code in events:
                if event_code not in inv_event_id:
                    continue

                label = inv_event_id[event_code]
                t_event = sample_idx / sfreq

                # La ventana termina en t_evento + epoch_duration (fin del epoch real)
                # y empieza window_size segundos antes, de modo que el epoch siempre
                # corresponde a [t_evento, t_evento + epoch_duration] sin importar
                # el tamaño de ventana (el resto sirve para calentar el filtro).
                t_end   = t_event + self._epoch_duration
                t_start = t_end - self._window_size

                # Si la ventana se sale por el inicio del raw, anclarla en 0
                if t_start < 0:
                    t_start = 0.0

                # Descartar ventanas que se salgan del final del registro
                if t_end > raw.times[-1]:
                    print(f"  Evento en {t_event:.2f}s: fin de ventana ({t_end:.2f}s) excede el raw ({raw.times[-1]:.2f}s), se omite.")
                    continue

                # 1. Recortar ventana
                raw_win = raw.copy().crop(tmin=t_start, tmax=t_end)

                # 2. Filtrar sólo la ventana
                raw_win = self._bandpass.process(raw_win)

                # 3. Tomar los últimos epoch_duration segundos
                data = raw_win.get_data()          # (n_ch, n_samples)
                if data.shape[1] < epoch_samples:
                    print(f"  Ventana demasiado corta ({data.shape[1]} muestras), se omite.")
                    continue

                epoch_data = data[:, -epoch_samples:]   # (n_ch, epoch_samples)
                all_X.append(epoch_data)
                all_Y.append(label)

        if not all_X:
            raise RuntimeError("No se extrajeron epochs. Revisa los archivos FIF y los nombres de anotaciones.")

        X       = np.array(all_X)          # (n_epochs, n_ch, n_samples)
        Y       = np.array(all_Y)
        classes = sorted(set(all_Y))

        return X, Y, classes

    def get_channel_names(self) -> List[str]:
        if not self._fif_paths:
            raise ValueError("No se han proporcionado archivos FIF.")

        raw = mne.io.read_raw_fif(self._fif_paths[0], preload=False, verbose=False)
        raw = raw.pick_types(eeg=True)
        return [ch.upper() for ch in raw.ch_names]
