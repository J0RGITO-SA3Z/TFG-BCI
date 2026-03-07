from typing import Dict, List, Optional, Sequence

import mne
import numpy as np
import os, sys

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from DataProvider.DataProvider import DataProvider
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
}

CLASS_NAMES = ["feet", "left_hand", "right_hand"]  # orden alfabético = orden real de LabelEncoder

def _raw_to_epochs(raw, tmin=0.0, tmax=4.0):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in CLASS_NAMES}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    return epochs

class FifDataProvider(DataProvider):
    def __init__(self, fif_paths: str | Sequence[str] = [], raw_pipeline: Optional[RawProcessorPipeline] = None) -> None:
        if isinstance(fif_paths, str):
            fif_paths = [fif_paths]
        self._fif_paths = list(fif_paths)
        self._raw_pipeline = raw_pipeline

    def add_fif_path(self, path: str) -> None:
        self._fif_paths.append(path)
    
    def remove_fif_path(self, path: str) -> None:
        self._fif_paths.remove(path)    

    def get_fif_paths(self) -> List[str]:
        return list(self._fif_paths)
    
    def get_data(self):
        all_epochs_list: List[np.ndarray] = []
        all_labels: List[str] = []

        for path in self._fif_paths:
            print(f"Cargando {path} ...")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

            if self._raw_pipeline is not None:
                raw = self._raw_pipeline.process(raw)

            single_epoch = _raw_to_epochs(raw)

            all_epochs_list.append(single_epoch)

        combined_epochs = mne.concatenate_epochs(all_epochs_list)
        
        X = combined_epochs.get_data()

        true_labels_numeric = combined_epochs.events[:, 2]
        inv_event_id = {v: k for k, v in combined_epochs.event_id.items()}
        true_labels = [inv_event_id[i] for i in true_labels_numeric]

        classes = sorted(set(true_labels))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[l] for l in true_labels], dtype=np.int64)

        return X, y, classes
   

    

"""

    def __init__(
        self,
        fif_paths: str | Sequence[str],
        event_map: Dict[str, str] | None = None,
        tmin: float = 0.0,
        tmax: float = 4.0,
        raw_pipeline: RawProcessorPipeline | None = None,
    ) -> None:
        if isinstance(fif_paths, str):
            fif_paths = [fif_paths]
        self._fif_paths = list(fif_paths)
        self._event_map = event_map
        self._tmin = tmin
        self._tmax = tmax
        self._raw_pipeline = raw_pipeline

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def fif_paths(self) -> List[str]:
        return list(self._fif_paths)

    @fif_paths.setter
    def fif_paths(self, value: str | Sequence[str]) -> None:
        if isinstance(value, str):
            value = [value]
        self._fif_paths = list(value)

    @property
    def event_map(self) -> Dict[str, str] | None:
        return self._event_map

    @event_map.setter
    def event_map(self, value: Dict[str, str] | None) -> None:
        self._event_map = value

    @property
    def raw_pipeline(self) -> RawProcessorPipeline | None:
        return self._raw_pipeline

    @raw_pipeline.setter
    def raw_pipeline(self, value: RawProcessorPipeline | None) -> None:
        self._raw_pipeline = value

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_data(self):
        all_epochs_data: List[np.ndarray] = []
        all_labels: List[str] = []

        for path in self._fif_paths:
            print(f"Cargando {path} ...")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

            if self._raw_pipeline is not None:
                raw = self._raw_pipeline.process(raw)

            epochs_data, labels = self._extract_epochs(raw)
            all_epochs_data.append(epochs_data)
            all_labels.extend(labels)

        X = np.concatenate(all_epochs_data, axis=0)
        labels_arr = np.array(all_labels)

        classes = sorted(set(labels_arr))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[l] for l in labels_arr], dtype=np.int64)

        print(f"  Shape X: {X.shape}  |  clases: {classes}")
        print(f"  Mapeo: {label_map}")

        return X, y, classes

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _extract_epochs(self, raw: mne.io.Raw):
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        if self._event_map is not None:
            event_id_filtered = {
                k: v for k, v in event_id.items() if k in self._event_map
            }
        else:
            event_id_filtered = event_id

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id_filtered,
            tmin=self._tmin,
            tmax=self._tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        epochs = epochs.copy().pick("eeg")

        inv_event_id = {v: k for k, v in epochs.event_id.items()}
        numeric_labels = epochs.events[:, 2]

        if self._event_map is not None:
            labels = [self._event_map[inv_event_id[n]] for n in numeric_labels]
        else:
            labels = [inv_event_id[n] for n in numeric_labels]

        epochs_data = epochs.get_data()  # (B, C, T)
        return epochs_data, labels

"""

