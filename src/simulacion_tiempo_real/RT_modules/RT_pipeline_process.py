from __future__ import annotations

import threading
import time
import os, sys
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
    """
    Clase encargada de ejecutar el pipeline de procesamiento en tiempo real. 
    Este pipeline se encarga de preprocesar, normalizar y clasificar los datos EEG recibidos en tiempo real desde el proceso de adquisición.
    El pipeline esta pensado para ejecutarse en un proceso separado en el cual se reciben las lecturas del EEG a través de un pipe
    y se envían las predicciones a través de otro pipe.

    Parameters
    ----------
    ea_matrix : np.ndarray
        Matriz precalculada para el alineamiento euclídeo (normalizacion).
    
    pretrained_model : ModelInterface
        Modelo de clasificación ya entrenado y listo para hacer predicciones.

    info : mne.Info
        Objeto mne.Info con la información de los canales y la frecuencia de muestreo necesario en el buffer circular
        para poder pasar los datos de np a "mne.raw" y así poder descartar canales no EEG y aplicar preprocesado. 
    
    channel_positions : dict
        Nombres de los canales y su posicion en el array dado por el EEG. Se necesita en el buffer circular para poder 
        pasar los datos de np a "mne.raw" y así poder descartar canales no EEG y aplicar preprocesado.

    channelNames : list
        Lista con los nombres de los canales en el orden que el modelo espera. 
        Se necesita para la interpolación espacial y reordenamiento de canales. (Normalizacion)
    
    stop_event : mp.Event
        Evento para indicar al proceso que debe parar. El proceso se mantendrá vivo hasta que este evento se active.
    """

    def __init__(self,ea_matrix,pretrained_model,info,channel_positions, stop_event, predecir_event) -> None:
        self.stop_event = stop_event
        self.predecir_event = predecir_event
        self.acquisition_thread = None

        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions

        self.buffer = buffer(info, channel_positions)

        eeg_picks = mne.pick_types(info, eeg=True, meg=False, eog=False, ecg=False, emg=False, misc=False, stim=False, exclude=[])
        eeg_channel_names = [info['ch_names'][idx] for idx in eeg_picks]

        if len(eeg_channel_names) == 0:
            raise ValueError("No se encontraron canales EEG en info['ch_names']")

        self.epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(matrix = ea_matrix),         # alineamiento euclídeo (EA)
            SpatialInterpolator(actual_channel_positions = eeg_channel_names),        # interpola/reordena canales a la topología objetivo 
        ])

    def run_loop(self,eeg_input_pipe, rt_interpreter_pipe):
        """
        Este método inicia el bucle principal del pipeline encargado de preprocesar, normalizar y clasificar los datos EEG en tiempo real.

        El pipeline recibe datos de eeg_input_pipe y envía las predicciones al proceso de interpretación en tiempo real.

        1-preprocesamiento: reduce el ruido y mejora la calidad de la señal EEG. Esta función únicamente descarta canales no EEG
        y aplica un filtro por frecuencia (entre 8 y 30 Hz) para quedarnos con las bandas de frecuencia relevantes para la tarea de MI.

        2-normalización: adaptar los datos EEG a la topología de canales y al formato que el modelo espera.
        En este caso solo se aplica alineamiento euclídeo (EA) con una matriz de transformación precalculada.

        3-clasificación: modelo de aprendizaje automático previamente entrenado. En este caso se utiliza una interfaz de modelo definida en
        model_interface.py.

        Para la recepción de los datos EEG se utiliza un hilo separado que escucha continuamente el pipe de entrada y va almacenando los datos en un buffer circular.
        El bucle se ejecuta indefinidamente hasta que se active `self.stop_event`.
        
        Parameters
        ----------
        eeg_input_pipe : multiprocessing.Connection
            Pipe desde el que se reciben los chunks de datos EEG provenientes
            del proceso de adquisición, junto con una marca de tiempo de cuándo se recibió el dato en el primer proceso.
            Con el formato (data, last_timestamp).

        rt_interpreter_pipe : multiprocessing.Connection
            Pipe de salida del RT_interpreter_process utilizado para enviar las predicciones del modelo
            junto con el número de la última muestra y la información temporal correspondiente.
            Formato enviado: (prediction, last_sample, last_timestamp).

        Notes
        -----
        El bucle de procesamiento se ejecuta aproximadamente cada 100 ms para
        generar predicciones en tiempo real.
        """

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
                rt_interpreter_pipe.send((prediction, last_sample, last_timestamp))

            time.sleep(max(0, next_time - time.perf_counter()))

        self.acquisition_thread.join()

    def acquisition_loop(self, pipe):
        """
        Bucle de adquisición encargado de recibir los datos EEG desde el proceso de adquisición.
        Pensado para ejecutarse en un hilo separado dentro del proceso del pipeline.

        Notes
        -----
        El bucle se ejecuta hasta que `self.stop_event` es activado.
        """
        while not self.stop_event.is_set():
            if pipe.poll(0.01):
                data,chunkSize ,last_timestamp = pipe.recv() # (channels, samples)
                print(f"timestamp {last_timestamp:.3f}")
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

        self.process = None

    @staticmethod
    def _process_target(ea_matrix, pretrained_model, info, channel_positions,
                        eeg_input_pipe, rt_interpreter_pipe, stop_event, predecir_event):

        pipeline = RT_pipeline(
            ea_matrix=ea_matrix,
            pretrained_model=pretrained_model,
            info=info,
            channel_positions=channel_positions,
            stop_event=stop_event,
            predecir_event=predecir_event,
        )

        pipeline.run_loop(eeg_input_pipe, rt_interpreter_pipe)

    def run_process(self, rt_interpreter_process):

        if self.process is not None and self.process.is_alive():
            return

        if rt_interpreter_process is None or getattr(rt_interpreter_process, "_send_pipe", None) is None:
            raise RuntimeError("RT_interpreter_process no esta iniciado o no tiene pipe de salida")

        self.stop_event.clear()
        self._eeg_input_parent_pipe, self._eeg_input_child_pipe = mp.Pipe()

        rt_interpreter_pipe = rt_interpreter_process._send_pipe

        self.process = mp.Process(
            target=self._process_target,
            args=(
                self.ea_matrix,
                self.pretrained_model,
                self.info,
                self.channel_positions,
                self._eeg_input_child_pipe,
                rt_interpreter_pipe,
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

        print("RT_pipeline_process detenido.")