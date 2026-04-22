import random
import socket
import sys, os
import time
from typing import List, Optional

import numpy as np
import winsound

from brainaccess.core.eeg_manager import EEGManager

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.DataProvider.DataProvider import DataProvider
from components.DataProvider.FifDataProvider import LABEL_MAP, _raw_to_epochs
from app.EEGRecorder import EEGRecorder
from app.tcp.eeg_live_server import EEGLiveServer, PORT
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from util.ventanaExperimentoVisual import ventanaExperimentoVisual


class ExperimentoVisualDataProvider(DataProvider):
    """
    DataProvider que ejecuta un experimento visual online
    """

    def __init__(self, puerto_COM, channelsConfig, numTrialsClase, lista, tmp_baseline_inicial = 20, tmp_baseline_epoch = 2, tmp_break = 2, tmp_im = 4):
        self._puerto_COM = puerto_COM
        self._channelsConfig = channelsConfig
        self._num_trials_clase = int(numTrialsClase)
        self._lista = list(lista)
        self.raw = None

        if self._num_trials_clase <= 0:
            raise ValueError("numTrialsClase debe ser mayor que 0")
        if not self._lista:
            raise ValueError("lista no puede estar vacia")

        self._tmp_baseline_inicial = tmp_baseline_inicial
        self._tmp_baseline_epoch = tmp_baseline_epoch
        self._tmp_break = tmp_break
        self._tmp_im = tmp_im

        self._raw_pipeline = RawProcessorPipeline([
        BandpassFilter(8.0, 30.0),
        AnnotationRenamer(LABEL_MAP),
        ])

        self._last_channel_names: Optional[List[str]] = None

    def _generar_lista(self, acciones, total):
        n = len(acciones)
        if n == 0 or total % n != 0:
            raise ValueError("El total debe ser multiplo del numero de acciones")

        lista = acciones * (total // n)
        random.shuffle(lista)
        return lista

    def _trial_to_text(self, trial):
        trial_upper = str(trial).upper()
        if trial_upper in ("IZQUIERDA", "LEFT_HAND"):
            return "<-"
        if trial_upper in ("DERECHA", "RIGHT_HAND"):
            return "->"
        if trial_upper in ("ABAJO", "FEET"):
            return "v"
        return str(trial)

    def _get_event_filter_names(self) -> List[str]:
        names = set(str(x) for x in self._lista)
        names.update(LABEL_MAP.get(str(x).upper(), str(x)) for x in self._lista)
        return sorted(names)

    def _ejecutar_experimento_visual(self):
        if self._puerto_COM is None:
            raise ValueError("puerto_COM no puede ser None")

        total_trials = self._num_trials_clase * len(self._lista)
        trials = self._generar_lista(self._lista, total_trials)

        eeg = EEGRecorder()
        live_server = None
        ventana = None
        raw = None

        with EEGManager() as mgr:
            eeg.configAndConect(
                mgr=mgr,
                COM_port=self._puerto_COM,
                channelConfig=self._channelsConfig,
                gain=8,
            )

            live_server = EEGLiveServer(
                ch_names=eeg.get_ch_names_ordered(),
                sfreq=eeg.get_sfreq(),
                ch_types=eeg.get_ch_types_ordered(),
                total_epochs=total_trials,
                initial_action="Empezando",
            )
            live_server.start()
            eeg.register_callback(live_server.newChunk)

            #Obtengo mi IP
            try:
                _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                _s.connect(("8.8.8.8", 80))
                local_ip = _s.getsockname()[0]
                _s.close()
            except Exception:
                local_ip = "127.0.0.1"

            print(f"\n{'='*52}")
            print(f"  Servidor EEG listo")
            print(f"  IP:     {local_ip}")
            print(f"  Puerto: {PORT}")
            print(f"{'='*52}")
            print("  Conecta el visualizador desde el otro portátil")
            print("  y pulsa INTRO cuando estés listo para empezar.")
            print(f"{'='*52}\n")
            input()

            ventana = ventanaExperimentoVisual()
            ventana.open()

            eeg.iniciarGrabacion()

            self._last_channel_names = [
                name.upper()
                for name, ch_type in zip(eeg.get_ch_names_ordered(), eeg.get_ch_types_ordered())
                if ch_type == "eeg"
            ]
            live_server.sendInfo(
                ch_names=eeg.get_ch_names_ordered(),
                ch_types=eeg.get_ch_types_ordered(),
                sfreq=eeg.get_sfreq(),
            )

            time.sleep(1)
            ventana.draw_text("Baseline")
            live_server.setAction("Baseline")
            eeg.anotar("INICIO_BASELINE")

            time.sleep(self._tmp_baseline_inicial - 5)
            winsound.Beep(1000, 500)
            time.sleep(5)

            for i, trial in enumerate(trials, start=1):
                trial_str = str(trial)
                live_server.setAction(trial_str)
                live_server.increaseEpoch()

                ventana.draw_text("+", current=i, total=total_trials)
                eeg.anotar("CROSS")
                time.sleep(self._tmp_baseline_epoch - 0.5)
                winsound.Beep(1000, 500)

                ventana.draw_text(self._trial_to_text(trial_str), current=i, total=total_trials)
                eeg.anotar(trial_str)

                time.sleep(self._tmp_im)
                ventana.draw_text("", current=i, total=total_trials)
                eeg.anotar("BLANK")

                time.sleep(self._tmp_break)

            print("Experimento terminado. Deteniendo grabacion...")
            raw = eeg.get_mne()
            eeg.detenerGrabacion()
            mgr.disconnect()

            live_server.stop()
            ventana.close()

        eeg.cerrarLibreria()
        return raw

    def get_data(self, fif_path=None):
        self.raw = self._ejecutar_experimento_visual()    

        if fif_path is not None:
            self.raw.save(fif_path, overwrite=True)

        if self._raw_pipeline is not None:
            processed_raw = self._raw_pipeline.process(self.raw.copy())

        annotations_names = self._get_event_filter_names()
        epochs = _raw_to_epochs(processed_raw, anotationsNames=annotations_names)

        X = epochs.get_data()

        true_labels_numeric = epochs.events[:, 2]
        inv_event_id = {v: k for k, v in epochs.event_id.items()}
        labels = np.array([inv_event_id[i] for i in true_labels_numeric])

        classes = sorted(set(labels))
        y = labels

        return X, y, classes

    def get_channel_names(self) -> List[str]:
        if self._last_channel_names is None:
            raise ValueError(
            "Los nombres de canales no estan disponibles aun. "
            "Ejecuta get_data() al menos una vez."
            )
        
        return list(self._last_channel_names)
    
    def get_raw(self):
        if self.raw is None:
            raise ValueError("Raw no disponible. Ejecuta get_data() primero.")
        return self.raw.copy()