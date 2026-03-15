from __future__ import annotations

from typing import Any

import threading
import time
import os, sys
import numpy as np
from real_time_raw_preprocesing_buffer import RealTimeRawPreprocessingBuffer as buffer
import multiprocessing as mp

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Data Processing ─────────────────────────────────────────────────────────────────────
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.EpochNormalizer import EpochNormalizer
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.ClassEventRemover import ClassEventRemover

class RT_pipeline:
    """
    Clase encargada de ejecutar el pipeline de procesamiento en tiempo real. 
    Este pipeline se encarga de preprocesar, normalizar y clasificar los datos EEG recibidos en tiempo real desde el proceso de adquisición.
    El pipeline esta pensado para ejecutarse en un proceso separado en el cual se reciben las lecturas del EEG a través de un pipe
    y se envían las predicciones a través de otro pipe.

    Se guarda

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

    def __init__(self,ea_matrix,pretrained_model,info,channel_positions,channelNames, stop_event) -> None:
        self.stop_event = stop_event
        self.acquisition_thread = None

        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions

        self.buffer = buffer(info, channel_positions)

        self.epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(matrix = ea_matrix),         # alineamiento euclídeo (EA)
            SpatialInterpolator(actual_channel_positions = channelNames),        # interpola/reordena canales a la topología objetivo 
        ])

    def run_loop(self,eeg_input_pipe, model_output_pipe):
        """
        Este método inicia el bucle principal del pipeline encargado de preprocesar, normalizar y clasificar los datos EEG en tiempo real.

        El pipeline recibe datos de eeg_input_pipe y devuelve las predicciones a través de model_output_pipe.

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

        model_output_pipe : multiprocessing.Connection
            Pipe utilizado para enviar las predicciones del modelo junto con el número de la última muestra y
            la información temporal correspondiente.
            Con el formato (prediction, last_sample, last_timestamp).

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
            #Aqui
            next_time += 0.15
            data, last_sample, last_timestamp = self.buffer.getData()
            data, _ = self.epoch_pipeline.process_np([data], [0])[0]
            prediction, probs = self.pretrained_model.predict_preprocessed(data)
            model_output_pipe.send((prediction, last_sample, last_timestamp))
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
                data = pipe.recv()  # (channels, samples)
                self.buffer.receiveData(data, data.shape[1])

class RT_pipeline_process:

    def __init__(self, ea_matrix, pretrained_model, info, channel_positions, channelNames):
        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions
        self.channelNames = channelNames

        self.stop_event = mp.Event()

        self.process = None

    @staticmethod
    def _process_target(ea_matrix, pretrained_model, info, channel_positions, channelNames,
                        eeg_input_pipe, model_output_pipe, stop_event):

        pipeline = RT_pipeline(
            ea_matrix=ea_matrix,
            pretrained_model=pretrained_model,
            info=info,
            channel_positions=channel_positions,
            channelNames=channelNames,
            stop_event=stop_event,
        )

        pipeline.run_loop(eeg_input_pipe, model_output_pipe)

        # Esperar hasta que pidan parar
        while not stop_event.is_set():
            pass

        pipeline.stop()

    def run_process(self, eeg_input_pipe, model_output_pipe):

        if self.process is not None and self.process.is_alive():
            return

        self.stop_event.clear()

        self.process = mp.Process(
            target=self._process_target,
            args=(
                self.ea_matrix,
                self.pretrained_model,
                self.info,
                self.channel_positions,
                self.channelNames,
                eeg_input_pipe,
                model_output_pipe,
                self.stop_event,
            ),
            daemon=True
        )

        self.process.start()

    def stop_process(self):

        self.stop_event.set()

        if self.process is not None:
            self.process.join()
            self.process = None