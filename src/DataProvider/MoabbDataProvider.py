"""
Proveedor de datos basado en datasets de MOABB.

Uso::

    provider = MoabbDataProvider("BNCI2014001", subject_idx=0)
    X, y, classes = provider.get_data()
"""

import numpy as np

import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004, BNCI2015_001
from moabb.paradigms import MotorImagery

from DataProvider.DataProvider import DataProvider
import os, sys

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from pretrainedModels.MiRepNet.utils.channel_list import BNCI2014001_chn_names, BNCI2014004_chn_names, BNCI2015001_chn_names, AlexMI_chn_names

moabb.set_log_level("ERROR")

_DATASET_MAP = {
    "BNCI2014001": BNCI2014_001,
    "BNCI2014004": BNCI2014_004,
    "BNCI2015001": BNCI2015_001,
}

class MoabbDataProvider(DataProvider):

    def __init__(
        self,
        dataset_name: str,
        subject_idx: int = 0,
        resample: float = 250.0,
        fmin: float = 8.0,
        fmax: float = 30.0,
    ) -> None:
        if dataset_name not in _DATASET_MAP:
            raise ValueError(
                f"Dataset '{dataset_name}' no soportado. "
                f"Opciones: {list(_DATASET_MAP)}"
            )
        self._dataset_name = dataset_name
        self._subject_idx = subject_idx
        self._resample = resample
        self._fmin = fmin
        self._fmax = fmax

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str) -> None:
        if value not in _DATASET_MAP:
            raise ValueError(
                f"Dataset '{value}' no soportado. Opciones: {list(_DATASET_MAP)}"
            )
        self._dataset_name = value

    @property
    def subject_idx(self) -> int:
        return self._subject_idx

    @subject_idx.setter
    def subject_idx(self, value: int) -> None:
        self._subject_idx = value

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_data(self):
        dataset = _DATASET_MAP[self._dataset_name]()
        subjects = dataset.subject_list
        subject_id = subjects[self._subject_idx]

        print(
            f"Cargando {self._dataset_name} — sujeto {subject_id} "
            f"({self._subject_idx + 1}/{len(subjects)})"
        )

        paradigm = MotorImagery(
            resample=self._resample, fmin=self._fmin, fmax=self._fmax
        )
        X, labels, _ = paradigm.get_data(dataset, subjects=[subject_id])

        classes = sorted(set(labels))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[l] for l in labels], dtype=np.int64)

        return X, y, classes
    
    def get_channel_names(self):
        if self.dataset_name == 'BNCI2014001':
            channels_names = BNCI2014001_chn_names
        elif self.dataset_name == 'BNCI2014004':
            channels_names = BNCI2014004_chn_names
        elif self.dataset_name == 'BNCI2014001-4':
            channels_names = BNCI2014001_chn_names
        elif self.dataset_name == 'AlexMI':
            channels_names = AlexMI_chn_names
        elif self.dataset_name =='BNCI2015001':
            channels_names = BNCI2015001_chn_names
    
        return channels_names
