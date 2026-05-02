import json
import numpy as np
import torch
import os
import sys
from sklearn.model_selection import train_test_split

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
PLAYER_PARAM_DIR = os.path.normpath(os.path.join(SRC_ROOT, "..", "recordings", "player_param"))

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

# ── Imports modelInterface ──────────────────────────────────────────────────────────
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

# ── Imports visualización ─────────────────────────────────────────────────────
from util.Performance_Viewer import PerformanceViewer

# ── Data Processing ─────────────────────────────────────────────────────────────────────
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.EuclideanAlignment import Calculate_EA_Matrix
from components.DataProvider.ExperimentoVisualDataProvider import ExperimentoVisualDataProvider


class Training_real_time:

    def __init__(self):
        self.channelsnames = None
        self.matrix = None
        self.modelo = None
        self.classes = None

    # Métodos Publicos de la clase  ─────────────────────────────────────────────────────────────────────
    def start(self, puerto_COM, channelsConfig, lista=["left_hand", "right_hand", "feet", "rest"], numTrialsClase=30, epochs=10, seed=42, fif_name=None):
        """
        Ejecuta un experimento visual con las clases pasadas en lista y con ello hace el fine tuning del modelo MiRepNet
        y calcula la matriz de alineamiento euclídeo (EA).
        En el caso de recibir un fif_name diferente que None guarda el experimento visual en un archivo .fif

        Parámetros:
            - puerto_COM: Puerto COM al que está conectado el dispositivo de adquisición EEG
            - channelsConfig: Configuración de canales a usar en el experimento visual
            - lista: Lista de clases a usar en el experimento visual (default: ["left_hand", "right_hand", "feet", "rest"])
            - numTrialsClase: Número de trials por clase a usar en el experimento visual (default: 30)
            - epochs: Número de épocas a usar en el fine-tuning del modelo (default: 10)
            - seed: Semilla para la reproducibilidad del experimento (default: 42)
            - fif_name: Nombre del archivo .fif donde se guardará la grabación del experimento visual (default: None, no se guarda)
        """
        if self._params_exist():
            resp = ""
            while resp not in ("s", "n"):
                resp = input("¿Cargar los últimos parámetros guardados? (s/n): ").strip().lower()
            if resp == "s":
                return self._load_params()

        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_provider = ExperimentoVisualDataProvider(
            puerto_COM=puerto_COM,
            channelsConfig=channelsConfig,
            numTrialsClase=numTrialsClase,
            lista=lista,
            tmp_baseline_inicial=20,
        )

        X, Y, classes = data_provider.get_data(fif_path=fif_name)
        self.channelsnames = data_provider.get_channel_names()
        self.classes = classes

        epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(),
            SpatialInterpolator(actual_channel_positions=self.channelsnames),
        ])

        self.modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, training_clases=lista)

        self.matrix = Calculate_EA_Matrix(X)
        X, Y = epoch_pipeline.process_np(X, Y, shuffle=False)

        historico = self.modelo.finetuning(X, Y, epochs=epochs)

        viewer = PerformanceViewer()
        viewer.summary(historico)

        self._save_params()

        return self.matrix, self.modelo

    # ------------------------------------------------------------------
    # Persistencia de parámetros del jugador
    # ------------------------------------------------------------------

    def _params_exist(self) -> bool:
        return (
            os.path.isfile(os.path.join(PLAYER_PARAM_DIR, "ea_matrix.npy"))
            and os.path.isfile(os.path.join(PLAYER_PARAM_DIR, "model_finetuned.pth"))
            and os.path.isfile(os.path.join(PLAYER_PARAM_DIR, "config.json"))
        )

    def _save_params(self) -> None:
        os.makedirs(PLAYER_PARAM_DIR, exist_ok=True)
        np.save(os.path.join(PLAYER_PARAM_DIR, "ea_matrix.npy"), self.matrix)
        self.modelo.save(os.path.join(PLAYER_PARAM_DIR, "model_finetuned.pth"))
        with open(os.path.join(PLAYER_PARAM_DIR, "config.json"), "w") as f:
            json.dump({"training_clases": list(self.classes)}, f)
        print(f"Parámetros guardados en: {PLAYER_PARAM_DIR}")

    def _load_params(self):
        with open(os.path.join(PLAYER_PARAM_DIR, "config.json")) as f:
            config = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matrix = np.load(os.path.join(PLAYER_PARAM_DIR, "ea_matrix.npy"))
        self.modelo = MiRepNetInterface(
            device=device,
            weight_path=os.path.join(PLAYER_PARAM_DIR, "model_finetuned.pth"),
            training_clases=config["training_clases"],
        )
        print(f"Parámetros cargados desde: {PLAYER_PARAM_DIR}")
        return self.matrix, self.modelo

    def getMatrix(self):
        return self.matrix

    def getModelo(self):
        return self.modelo
