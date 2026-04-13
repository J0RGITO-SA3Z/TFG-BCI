from __future__ import annotations

import time
import os, sys
import numpy as np
import threading
from typing import Optional
import mne

import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.utils.acquisition import EEGData_roll

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler
from raw_processing.CARReference import CARReference
from raw_processing.ICAProcessor import ICAProcessor
from raw_processing.AnnotationRenamer import AnnotationRenamer

class RealTimeRawPreprocessingBuffer:
    def __init__(self,info, channels_indexes ,raw_pipeline = None,ventana_procesado = 10, frecuecia_muestreo: int = 250, venetana_res: int = 4):
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()
        
        self.info = info
        self.channels_indexes = channels_indexes
        self.data = EEGData_roll(info, self.lock, ventana_procesado*frecuecia_muestreo)
        self.last_time: Optional[float] = None
        self.total_samples_received = 0

        if raw_pipeline is not None:
            self._raw_pipeline = raw_pipeline
        else:
            self._raw_pipeline = RawProcessorPipeline([
                # NotchFilter(50.0),
                BandpassFilter(8.0, 30.0),
                #AnnotationRenamer(LABEL_MAP),
                #CARReference(),
                # Resampler(250),
                # ICAProcessor(),
            ])

        self.ventana_procesado = ventana_procesado
        self.frecuecia_muestreo = frecuecia_muestreo
        self.venetana_res = venetana_res

    
    def get_last_sample(self, raw) -> int:
        """
        Devuelve el último valor del canal 'Sample'
        correspondiente a la última muestra registrada.
        """
        if "Sample" not in raw.ch_names:
            return self.total_samples_received

        sample_data = raw.get_data(picks=["Sample"])

        return int(sample_data[0, -1])

    def getData(self):
        last_timestamp = None

        with self.lock2:
            last_timestamp = self.last_time
            self.data.convert_to_mne(
                channels_indexes=list(self.channels_indexes.values()),
                annotations=False,
            )

            raw_full = self.data.mne_raw.copy()

        last_sample_number = self.get_last_sample(raw_full)
        raw = self._raw_pipeline.process(raw_full)

        n_samples = self.venetana_res * self.frecuecia_muestreo
        eeg_data = raw.get_data()[:, -n_samples:]

        n_samples = self.venetana_res * self.frecuecia_muestreo

        if eeg_data.shape[1] < n_samples:
            raise RuntimeError(
                f"getData: se esperaban {n_samples} muestras pero sólo hay "
                f"{eeg_data.shape[1]}. ¿Buffer insuficientemente lleno o sfreq incorrecto?"
            )

        return eeg_data, last_sample_number, last_timestamp

    def receiveData(self, chunk: np.ndarray, chunk_size: int,timestamp: Optional[float] = None):
        if timestamp is None:
                timestamp = time.perf_counter()

        with self.lock2:
            self.data.data = np.roll(self.data.data, -chunk_size, axis=1)
            self.data.data[:, -chunk_size:] = np.array(chunk)

            self.last_time = timestamp
            self.total_samples_received += chunk_size
