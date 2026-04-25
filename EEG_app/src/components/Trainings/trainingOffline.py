import os
import sys
from typing import Sequence

import torch
from sklearn.model_selection import train_test_split

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

# -- Modelo ------------------------------------------------------------------
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

# -- Visualizacion -----------------------------------------------------------
from util.Performance_Viewer import PerformanceViewer

# -- Proveedor de datos ------------------------------------------------------
from EEG_app.src.components.DataProvider.FifDataProvider import FifDataProvider

# -- Data Processing ---------------------------------------------------------
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.EuclideanAlignment import Calculate_EA_Matrix


class Training_offline:
	"""Entrenamiento offline basado en uno o varios archivos FIF."""

	def __init__(self):
		self.channelsnames = None
		self.matrix = None
		self.modelo = None
		self.history = None
		self.classes = None

	def start(
		self,
		fif_paths: str | Sequence[str],
		lista = ["left_hand", "right_hand", "feet", "rest"],
		epochs: int = 10,
		seed: int = 42,
		validation_split: float = 0,
	):
		"""
		Entrena MiRepNet usando datos provenientes de uno o varios archivos FIF.

		Parameters
		----------
		fif_paths : str | Sequence[str]
			Ruta de un FIF o lista de rutas FIF.
		lista : list[str]
			Clases usadas para entrenar (y para filtrar anotaciones del provider).
		epochs : int
			Numero de epocas de fine-tuning.
		seed : int
			Semilla para reproducibilidad.
		validation_split : float
			Fraccion de validacion para train_test_split.
		"""
		torch.manual_seed(seed)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		data_provider = FifDataProvider(
			fif_paths=fif_paths,
			annotations_names=lista,
		)

		X, y, classes = data_provider.get_data()
		self.channelsnames = data_provider.get_channel_names()
		self.classes = classes

		if len(X) == 0:
			raise ValueError("No se han generado epochs desde los FIF proporcionados")

		# Matriz de referencia para EA en tiempo real/inferencia.
		self.matrix = Calculate_EA_Matrix(X)

		epoch_pipeline = EpochProcessorPipeline([
			EuclideanAlignment(self.matrix),
			SpatialInterpolator(actual_channel_positions=self.channelsnames),
		])

		X, y = epoch_pipeline.process_np(X, y, shuffle=False)

		self.modelo = MiRepNetInterface(
			device=device,
			weight_path=WEIGHT_PATH,
			training_clases=lista,
		)

		if 0.0 < validation_split < 1.0:
			X_train, X_val, y_train, y_val = train_test_split(
				X,
				y,
				test_size=validation_split,
				random_state=seed,
				stratify=y,
			)
			self.history = self.modelo.finetuning(X_train, y_train, X_val, y_val, epochs=epochs)
		else:
			self.history = self.modelo.finetuning(X, y, epochs=epochs)

		viewer = PerformanceViewer()
		viewer.summary(self.history)

		return self.matrix, self.modelo

	def getMatrix(self):
		return self.matrix

	def getModelo(self):
		return self.modelo

	def getHistory(self):
		return self.history

	def getChannelNames(self):
		return self.channelsnames
