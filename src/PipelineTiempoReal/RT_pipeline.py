from __future__ import annotations

from typing import Any

import threading
import time
import os, sys
import numpy as np
import real_time_raw_preprocesing_buffer

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

    def __init__(self,ea_matrix,pretrained_model,info,channel_positions,channelNames) -> None:
        self.running = threading.Event()
        self.acquisition_thread = None
        self.processing_thread = None

        self.ea_matrix = ea_matrix
        self.pretrained_model = pretrained_model
        self.info = info
        self.channel_positions = channel_positions

        self.buffer = real_time_raw_preprocesing_buffer.RealTimeRawPreprocessingBuffer(info, channel_positions)

        self.epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(matrix = ea_matrix),         # alineamiento euclídeo (EA)
            SpatialInterpolator(actual_channel_positions = channelNames),        # interpola/reordena canales a la topología objetivo 
        ])

    def run_loop(self,eeg_input_pipe, model_output_pipe):
        if not self.running.is_set():
            self.running.set()
        else:
            print("Pipeline already running.")
            return
        
        self.acquisition_thread = threading.Thread(
            target=self.acquisition_loop,
            args=(eeg_input_pipe,),
            daemon=True
        )

        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            args=(model_output_pipe,),
            daemon=True
        )

        self.acquisition_thread.start()
        self.processing_thread.start()

    def processing_loop(self, model_output_pipe):
        next_time = time.perf_counter()

        while self.running.is_set():
            next_time += 0.1
            data, last_sample, last_timestamp = self.buffer.getData()
            data, _ = self.epoch_pipeline.process_np([data], [0])[0]
            prediction, probs = self.pretrained_model.predict_preprocessed(data)
            model_output_pipe.send((prediction, last_sample, last_timestamp))
            time.sleep(max(0, next_time - time.perf_counter()))

    def acquisition_loop(self, pipe):
        while self.running.is_set():
            if pipe.poll(0.01):
                data = pipe.recv()  # (channels, samples)
                self.buffer.receiveData(data, data.shape[1])

    def stop(self):
        self.running.clear()
