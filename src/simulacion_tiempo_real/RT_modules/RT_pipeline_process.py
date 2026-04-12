from __future__ import annotations

import threading
import time
import os, sys
import tempfile
import numpy as np
import multiprocessing as mp
import mne

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from RT_modules.real_time_raw_preprocesing_buffer import RealTimeRawPreprocessingBuffer as buffer

# ── Data Processing ─────────────────────────────────────────────────────────────────────
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment

class RT_pipeline:

    def __init__(self,ea_matrix,pretrained_model,info,channel_positions, stop_event, predecir_event) -> None:
        self.stop_event = stop_event
        self.predecir_event = predecir_event
        self.acquisition_thread = None

        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions

        self.buffer = buffer(info, channel_positions, frecuecia_muestreo=int(info['sfreq']))

        eeg_picks = mne.pick_types(info, eeg=True, meg=False, eog=False, ecg=False, emg=False, misc=False, stim=False, exclude=[])
        eeg_channel_names = [info['ch_names'][idx] for idx in eeg_picks]

        if len(eeg_channel_names) == 0:
            raise ValueError("No se encontraron canales EEG en info['ch_names']")

        self.epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(matrix = ea_matrix),         # alineamiento euclídeo (EA)
            SpatialInterpolator(actual_channel_positions = eeg_channel_names),        # interpola/reordena canales a la topología objetivo 
        ])

    def run_loop(self, eeg_input_pipe, interpreter_send_pipe):
        self.acquisition_thread = threading.Thread(
            target=self.acquisition_loop,
            args=(eeg_input_pipe,),
            daemon=True
        )

        self.acquisition_thread.start()

        next_time = time.perf_counter()

        while not self.stop_event.is_set():
            next_time += 0.15
            if self.predecir_event.is_set():
                data, last_sample, last_timestamp = self.buffer.getData()
                data = np.expand_dims(data, axis=0)  # (1, C, T)
                data, _ = self.epoch_pipeline.process_np(data, [0])
                data = data[0]
                prediction, probs = self.pretrained_model.predict(data)
                interpreter_send_pipe.send((prediction, last_sample, last_timestamp))

            time.sleep(max(0, next_time - time.perf_counter()))

        self.acquisition_thread.join()

    def acquisition_loop(self, pipe):
        while not self.stop_event.is_set():
            if pipe.poll(0.01):
                data,chunkSize ,last_timestamp = pipe.recv() # (channels, samples)
                self.buffer.receiveData(data, chunkSize, last_timestamp)

class RT_pipeline_process:

    def __init__(self, ea_matrix, pretrained_model, info, channel_positions):
        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions

        self.stop_event = mp.Event()
        self.predecir_event = mp.Event()
        self.predecir_event.clear()
        self._eeg_input_parent_pipe = None
        self._eeg_input_child_pipe = None
        self._model_weight_tmp = None

        self.process = None

    @staticmethod
    def _process_target(pipeline_params, eeg_input_pipe, interpreter_send_pipe, stop_event, predecir_event):

        from model_interface.MiRepNetInterface import MiRepNetInterface
        pretrained_model = MiRepNetInterface(**pipeline_params["model_params"])

        pipeline = RT_pipeline(
            ea_matrix=pipeline_params["ea_matrix"],
            pretrained_model=pretrained_model,
            info=pipeline_params["info"],
            channel_positions=pipeline_params["channel_positions"],
            stop_event=stop_event,
            predecir_event=predecir_event,
        )

        pipeline.run_loop(eeg_input_pipe, interpreter_send_pipe)

    def run_process(self, rt_interpreter_process):

        if self.process is not None and self.process.is_alive():
            return

        if rt_interpreter_process is None or getattr(rt_interpreter_process, "_send_pipe", None) is None:
            raise RuntimeError("RT_interpreter_process no esta iniciado o no tiene pipe de salida")

        # Guardar pesos en un fichero temporal y obtener los params del constructor
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        tmp.close()
        self._model_weight_tmp = tmp.name
        self.pretrained_model.save(self._model_weight_tmp)

        pipeline_params = {
            "ea_matrix":        self.ea_matrix,
            "model_params":     self.pretrained_model.get_constructor_params(self._model_weight_tmp),
            "info":             self.info,
            "channel_positions": self.channel_positions,
        }

        self.stop_event.clear()
        self._eeg_input_parent_pipe, self._eeg_input_child_pipe = mp.Pipe()

        self.process = mp.Process(
            target=self._process_target,
            args=(
                pipeline_params,
                self._eeg_input_child_pipe,
                rt_interpreter_process._send_pipe,
                self.stop_event,
                self.predecir_event,
            ),
            daemon=True
        )

        self.process.start()

    def sendData(self, data, dataSize):
        if self.process is None or not self.process.is_alive() or self._eeg_input_parent_pipe is None:
            raise RuntimeError("RT_pipeline_process no esta iniciado")

        timestamp = time.perf_counter()
        self._eeg_input_parent_pipe.send((data, dataSize,timestamp))

    def set_predecir(self, activo: bool):
        if activo:
            self.predecir_event.set()
        else:
            self.predecir_event.clear()

    def activar_predecir(self):
        self.predecir_event.set()

    def desactivar_predecir(self):
        self.predecir_event.clear()

    def stop_process(self):
        print("Deteniendo RT_pipeline_process...")
        self.stop_event.set()

        if self.process is not None:
            self.process.join()
            self.process = None

        if self._eeg_input_parent_pipe is not None:
            self._eeg_input_parent_pipe.close()
            self._eeg_input_parent_pipe = None

        if self._eeg_input_child_pipe is not None:
            self._eeg_input_child_pipe.close()
            self._eeg_input_child_pipe = None

        if self._model_weight_tmp is not None and os.path.exists(self._model_weight_tmp):
            os.remove(self._model_weight_tmp)
            self._model_weight_tmp = None

        print("RT_pipeline_process detenido.")