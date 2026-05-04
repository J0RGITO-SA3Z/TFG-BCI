import random
import socket
import sys, os
import time
from typing import List, Optional

import winsound

from brainaccess.core.eeg_manager import EEGManager

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.DataProvider.DataProvider import DataProvider
from components.DataProvider.FifDataProvider import LABEL_MAP, _raw_to_epochs, _extract_labels
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator
from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from app.EEGRecorder import EEGRecorder
from app.tcp.eeg_live_server import EEGLiveServer, PORT
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from util.ventanaExperimentoVisual import ventanaExperimentoVisual
from components.RawProcessing.NotchFilter import NotchFilter


class ExperimentoVisualDataProvider(DataProvider):
    """
    DataProvider que ejecuta un experimento visual online
    """

    def __init__(
        self,
        puerto_COM,
        channelsConfig,
        numTrialsClase,
        lista,
        tmp_baseline_inicial=20,
        tmp_baseline_epoch=2,
        tmp_break=2,
        tmp_im=4,
        raw_pipeline_detection: Optional[RawProcessorPipeline] = None,
        raw_pipeline_final: Optional[RawProcessorPipeline] = None,
        use_bad_channel_interpolator: bool = False,
        interpolate_bad_channels: bool = True,
        min_epochs_per_class: Optional[int] = None,
    ):
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
        self._use_bad_channel_interpolator = use_bad_channel_interpolator
        self._interpolate_bad_channels = interpolate_bad_channels
        self._min_epochs_per_class = min_epochs_per_class
        self._bad_channel_interpolator: Optional[BadChannelInterpolator] = None

        _default_pipeline_detection = RawProcessorPipeline([
            NotchFilter(50),
            BandpassFilter(1.0, 40.0),
            AnnotationRenamer(LABEL_MAP),
        ])
        _default_pipeline_final = RawProcessorPipeline([
            NotchFilter(50),
            BandpassFilter(8.0, 30.0),
            AnnotationRenamer(LABEL_MAP),
        ])

        self._raw_pipeline_detection = raw_pipeline_detection if raw_pipeline_detection is not None else _default_pipeline_detection
        self._raw_pipeline_final = raw_pipeline_final if raw_pipeline_final is not None else _default_pipeline_final

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

            eeg.iniciarGrabacion()

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

    def _build_bad_channel_interpolator(self) -> BadChannelInterpolator:
        return BadChannelInterpolator(
            channels_max=3,
            print_history=False,
            actual_channel_positions=self._last_channel_names,
            detectors=[
                AmplitudeThresholdDetector(threshold=100),
                VarianceDetector(threshold=1000.0, dead_threshold=2),
            ],
        )

    def get_data(self, fif_path=None):
        self.raw = self._ejecutar_experimento_visual()

        self._last_channel_names = [
            name.upper()
            for name, ch_type in zip(self.raw.ch_names, self.raw.get_channel_types())
            if ch_type == "eeg"
        ]

        if fif_path is not None:
            self.raw.save(fif_path, overwrite=True)

        annotations_names = self._get_event_filter_names()

        if self._use_bad_channel_interpolator:
            self._bad_channel_interpolator = self._build_bad_channel_interpolator()
            X, y = self._get_data_two_pass(self.raw, annotations_names)
        else:
            X, y = self._get_data_simple(self.raw, annotations_names)

        classes = sorted(set(y))
        return X, y, classes

    def _get_data_simple(self, raw, annotations_names):
        processed_raw = self._raw_pipeline_final.process(raw.copy())
        epochs = _raw_to_epochs(processed_raw, anotationsNames=annotations_names)
        return epochs.get_data(), _extract_labels(epochs)

    def _get_data_two_pass(self, raw, annotations_names):
        interp = self._bad_channel_interpolator

        # Paso 1: detección
        raw_det = self._raw_pipeline_detection.process(raw.copy())
        epochs_det = _raw_to_epochs(raw_det, anotationsNames=annotations_names)
        bad_channels, discarded = interp.detect_only(epochs_det.get_data())
        print(
            f"  Detección: {sum(bool(ch) for ch in bad_channels)} epochs con canales malos, "
            f"{len(discarded)} epochs descartados."
        )

        # Paso 2: procesado final
        raw_final = self._raw_pipeline_final.process(raw.copy())
        epochs_final = _raw_to_epochs(raw_final, anotationsNames=annotations_names)
        X_final = epochs_final.get_data()
        y_final = _extract_labels(epochs_final)

        if self._min_epochs_per_class is not None:
            discarded_set = set(discarded)
            counts: dict = {}
            for i, label in enumerate(y_final):
                if i not in discarded_set:
                    counts[label] = counts.get(label, 0) + 1
            if counts and min(counts.values()) < self._min_epochs_per_class:
                print(
                    f"  Mínimo de epochs por clase no alcanzado tras eliminación "
                    f"({min(counts.values())} < {self._min_epochs_per_class}). "
                    f"Omitiendo eliminación de malos."
                )
                return X_final, y_final

        return interp.apply_detected(X_final, y_final, interpolate=self._interpolate_bad_channels)

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