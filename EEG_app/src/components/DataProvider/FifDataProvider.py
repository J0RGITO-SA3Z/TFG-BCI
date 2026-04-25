from typing import Dict, List, Optional, Sequence

import mne
import numpy as np
import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from components.DataProvider.DataProvider import DataProvider
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator

# ─────── Imports pipeline ─────────────────────────────────────────────────────
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.NotchFilter import NotchFilter
from components.RawProcessing.Resampler import Resampler
from components.RawProcessing.CARReference import CARReference
from components.RawProcessing.ICAProcessor import ICAProcessor
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer

LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
    "DESCANSO":  "rest",
}

def _raw_to_epochs(raw, tmin=0.0, tmax=4.0, anotationsNames=["left_hand", "right_hand", "feet"]):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in anotationsNames}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    return epochs

def _extract_labels(epochs: mne.Epochs) -> np.ndarray:
    true_labels_numeric = epochs.events[:, 2]
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    return np.array([inv_event_id[i] for i in true_labels_numeric])


class FifDataProvider(DataProvider):
    """Carga ficheros .fif y devuelve (X, labels, classes).

    Modo simple (sin `bad_channel_interpolator`):
        Aplica `raw_pipeline_detection` y epoquiza directamente.

    Modo dos pasos (con `bad_channel_interpolator`):
        1. Copia del raw → `raw_pipeline_detection` → epochs → `detect_only`.
        2. Copia del raw → `raw_pipeline_final`     → epochs → `apply_detected`.

    Parameters
    ----------
    raw_pipeline_detection :
        Pipeline aplicado en la fase de detección de canales malos.
        Si se omite se usa un pipeline por defecto (BandpassFilter + AnnotationRenamer).
    raw_pipeline_final :
        Pipeline aplicado a los datos que se devuelven al caller.
        Solo se usa cuando se proporciona `bad_channel_interpolator`.
        Si se omite se usa `raw_pipeline_detection`.
    bad_channel_interpolator :
        Instancia de `BadChannelInterpolator` que realiza la detección
        (fase 1) y la interpolación/descarte (fase 2).
    """

    def __init__(
        self,
        fif_paths: str | Sequence[str] = [],
        raw_pipeline_detection: Optional[RawProcessorPipeline] = None,
        raw_pipeline_final: Optional[RawProcessorPipeline] = None,
        bad_channel_interpolator: Optional[BadChannelInterpolator] = None,
        interpolate_bad_channels: bool = True,
        annotations_names=["left_hand", "right_hand", "feet"],
    ) -> None:
        if isinstance(fif_paths, str):
            fif_paths = [fif_paths]

        self._fif_paths = list(fif_paths)
        self._annotations_names = annotations_names
        self._bad_channel_interpolator = bad_channel_interpolator
        self._interpolate_bad_channels = interpolate_bad_channels

        _default_pipeline_detection = RawProcessorPipeline([
            BandpassFilter(1, 40.0),
            AnnotationRenamer(LABEL_MAP),
        ])

        _default_pipeline_final = RawProcessorPipeline([
            BandpassFilter(8, 30.0),
            AnnotationRenamer(LABEL_MAP),
        ])

        self._raw_pipeline_detection = raw_pipeline_detection if raw_pipeline_detection is not None else _default_pipeline_detection
        # Si no se especifica pipeline final, se reutiliza el de detección
        self._raw_pipeline_final = raw_pipeline_final if raw_pipeline_final is not None else _default_pipeline_final

    # ── Gestión de rutas ───────────────────────────────────────────────────────

    def add_fif_path(self, path: str) -> None:
        self._fif_paths.append(path)

    def remove_fif_path(self, path: str) -> None:
        self._fif_paths.remove(path)

    def get_fif_paths(self) -> List[str]:
        return list(self._fif_paths)

    # ── get_data ───────────────────────────────────────────────────────────────

    def get_data(self):
        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for path in self._fif_paths:
            print(f"Cargando {path} ...")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

            if self._bad_channel_interpolator is not None:
                X, y = self._get_data_two_pass(raw)
            else:
                X, y = self._get_data_simple(raw)

            all_X.append(X)
            all_y.append(y)

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        classes = sorted(set(y_combined))

        return X_combined, y_combined, classes

    def _get_data_simple(self, raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
        """Pipeline único: detección → epoquización directa."""
        if self._raw_pipeline_final is not None:
            raw = self._raw_pipeline_final.process(raw)

        epochs = _raw_to_epochs(raw, anotationsNames=self._annotations_names)
        return epochs.get_data(), _extract_labels(epochs)

    def _get_data_two_pass(self, raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
        """Dos pasos: detección sobre pipeline 1, interpolación sobre pipeline 2.

        Paso 1 — Detección:
            Se aplica `raw_pipeline_detection` a una copia del raw para
            obtener epochs "sucios". `detect_only` identifica los canales
            malos y los epochs a descartar y guarda esa información.

        Paso 2 — Procesado final:
            Se aplica `raw_pipeline_final` a la copia original del raw.
            `apply_detected` interpola los canales malos detectados en el
            paso 1 y descarta los epochs marcados.
        """
        interp = self._bad_channel_interpolator

        # ── Paso 1: detección ──────────────────────────────────────────────
        raw_det = raw.copy()
        if self._raw_pipeline_detection is not None:
            raw_det = self._raw_pipeline_detection.process(raw_det)

        epochs_det = _raw_to_epochs(raw_det, anotationsNames=self._annotations_names)
        X_det = epochs_det.get_data()

        bad_channels, discarded = interp.detect_only(X_det)
        print(
            f"  Detección: {sum(bool(ch) for ch in bad_channels)} epochs con canales malos, "
            f"{len(discarded)} epochs descartados."
        )

        # ── Paso 2: procesado final ────────────────────────────────────────
        raw_final = raw.copy()
        if self._raw_pipeline_final is not None:
            raw_final = self._raw_pipeline_final.process(raw_final)

        epochs_final = _raw_to_epochs(raw_final, anotationsNames=self._annotations_names)
        X_final = epochs_final.get_data()
        y_final = _extract_labels(epochs_final)

        X_processed, y_processed = interp.apply_detected(X_final, y_final, interpolate=self._interpolate_bad_channels)

        return X_processed, y_processed

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def get_channel_names(self) -> List[str]:
        if not self._fif_paths:
            raise ValueError("No se han proporcionado archivos FIF para obtener los nombres de los canales.")

        raw = mne.io.read_raw_fif(self._fif_paths[0], preload=False, verbose=False)
        raw = raw.pick_types(eeg=True)

        return [a.upper() for a in raw.ch_names]

    def get_info(self) -> mne.Info:
        if not self._fif_paths:
            raise ValueError("No se han proporcionado archivos FIF.")
        raw = mne.io.read_raw_fif(self._fif_paths[0], preload=False, verbose=False)
        raw = raw.pick_types(eeg=True)
        return raw.info
