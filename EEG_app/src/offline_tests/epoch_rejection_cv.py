"""
Compara el accuracy con y sin epoch rejection para cada sujeto usando K-Fold CV.

Condiciones evaluadas (5 en total):
  1. Sin rechazo                     — filtro 8-30 Hz
  2. Rechazo + interp  detect=1-40   — detect=1-40, final=8-30
  3. Rechazo + interp  detect=8-30   — detect=8-30, final=8-30
  4. Solo rechazo      detect=1-40   — detect=1-40, final=8-30
  5. Solo rechazo      detect=8-30   — detect=8-30, final=8-30

Las condiciones con rechazo equilibran el número de trials por clase tras eliminar epochs malos.
"""
import os
import sys
import glob
import random
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

SRC_ROOT     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
RECORDINGS   = os.path.abspath(os.path.join(SRC_ROOT, "..", "recordings", "experimento_visual"))

for _p in [PROJECT_ROOT, SRC_ROOT, MIREPNET_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.DataProvider.FifDataProvider import FifDataProvider, LABEL_MAP
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator
from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

N_FOLDS         = 5
TRAIN_SIZE      = 0.15
SEED            = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS     = ["left_hand", "right_hand"]
MIN_PER_CLASS   = 4   # mínimo de epochs por clase por fold para que el experimento tenga sentido


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_subject_files(subject_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(subject_dir, "*_raw.fif")))


def _discover_subjects() -> dict[str, list[str]]:
    subjects = {}
    if not os.path.isdir(RECORDINGS):
        raise FileNotFoundError(f"No se encontró el directorio de grabaciones: {RECORDINGS}")
    for entry in sorted(os.listdir(RECORDINGS)):
        subject_dir = os.path.join(RECORDINGS, entry)
        if os.path.isdir(subject_dir) and entry.startswith("suj"):
            files = _get_subject_files(subject_dir)
            if files:
                subjects[entry] = files
    return subjects


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    channel_names: list[str],
) -> float:
    """Crea un modelo fresco, lo finetunea y devuelve el accuracy en test."""
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=ANNOTATIONS,
    )

    X_tr, y_tr = epoch_pipeline.process_np(X_train, y_train, shuffle=False)
    X_te, y_te = epoch_pipeline.process_np(X_test,  y_test,  shuffle=False)

    modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)
    preds, _ = modelo.predict_batch(X_te)
    return float(accuracy_score(y_te, preds))


def _kfold_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
) -> tuple[float, float]:
    """N_FOLDS repeticiones de split estratificado con TRAIN_SIZE de entrenamiento.

    Devuelve (media_acc, desv_típica_acc).
    """
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, train_size=TRAIN_SIZE, random_state=SEED)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        acc = _evaluate(X_tr, y_tr, X_te, y_te, channel_names)
        accs.append(acc)
        print(f"      Fold {fold + 1}/{N_FOLDS}: acc={acc:.4f}")

    return float(np.mean(accs)), float(np.std(accs))


def _has_enough_samples(y: np.ndarray) -> bool:
    """Comprueba que cada clase tenga suficientes muestras para N_FOLDS folds."""
    classes, counts = np.unique(y, return_counts=True)
    return len(classes) == len(ANNOTATIONS) and all(c >= N_FOLDS * MIN_PER_CLASS for c in counts)


def _balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Submuestreo aleatorio hasta igualar el número de trials de cada clase."""
    rng = np.random.RandomState(SEED)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    keep = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        keep.append(rng.choice(idx, size=min_count, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


# ── Pipelines ──────────────────────────────────────────────────────────────────

def _pipeline_no_rejection() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


def _pipeline_detection_broad() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(1, 40.0), AnnotationRenamer(LABEL_MAP)])


def _pipeline_detection_narrow() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


def _pipeline_final() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


def _bad_channel_interpolator(channel_names: list[str]) -> BadChannelInterpolator:
    return BadChannelInterpolator(
        channels_max=3,
        print_history=False,
        actual_channel_positions=channel_names,
        detectors=[
            AmplitudeThresholdDetector(threshold=100),
            VarianceDetector(threshold=1000.0, dead_threshold=2),
        ],
    )


# ── Lógica principal ───────────────────────────────────────────────────────────

def _run_rejection_condition(
    fif_paths: list[str],
    channel_names: list[str],
    interpolate: bool,
    detection_pipeline: RawProcessorPipeline,
    n_total: int,
) -> tuple[float | None, float | None, int]:
    """Carga datos con rechazo, equilibra clases y evalúa con K-Fold CV.

    Devuelve (mean_acc, std_acc, n_discarded).
    """
    interp = _bad_channel_interpolator(channel_names)
    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=detection_pipeline,
        raw_pipeline_final=_pipeline_final(),
        bad_channel_interpolator=interp,
        interpolate_bad_channels=interpolate,
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()
    n_discarded = n_total - len(X)

    if not _has_enough_samples(y):
        print(f"    AVISO: tras rechazo solo quedan {len(X)} epochs. Saltando evaluación.")
        return None, None, n_discarded

    X, y = _balance_classes(X, y)
    classes, counts = np.unique(y, return_counts=True)
    print(f"    Balance: {dict(zip(classes, counts))}  (total={len(y)})")
    mean, std = _kfold_evaluate(X, y, channel_names)
    print(f"    → {mean:.4f} ± {std:.4f}  (eliminados: {n_discarded}/{n_total})")
    return mean, std, n_discarded


def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    print(f"\n{'─' * 62}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)  —  {N_FOLDS}-Fold CV estratificado")
    print(f"{'─' * 62}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    result = {
        "sujeto":                    subj_name,
        "acc_sin_mean":              None, "acc_sin_std":              None,
        "acc_interp_broad_mean":     None, "acc_interp_broad_std":     None,
        "acc_interp_narrow_mean":    None, "acc_interp_narrow_std":    None,
        "acc_solo_broad_mean":       None, "acc_solo_broad_std":       None,
        "acc_solo_narrow_mean":      None, "acc_solo_narrow_std":      None,
        "eliminados_broad":          None,
        "eliminados_narrow":         None,
        "total":                     None,
    }

    # ── [1/5] Sin rechazo ─────────────────────────────────────────────────
    print("  [1/5] Sin epoch rejection (8-30 Hz)...")
    provider_plain = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_no_rejection(),
        annotations_names=ANNOTATIONS,
    )
    X_all, y_all, _ = provider_plain.get_data()
    n_total = len(X_all)
    result["total"] = n_total

    if not _has_enough_samples(y_all):
        print(f"  AVISO: {n_total} epochs insuficientes para {N_FOLDS} folds. Saltando sujeto.")
        return result

    mean_plain, std_plain = _kfold_evaluate(X_all, y_all, channel_names)
    result["acc_sin_mean"] = mean_plain
    result["acc_sin_std"]  = std_plain
    print(f"    → {mean_plain:.4f} ± {std_plain:.4f}")

    # ── [2/5] Rechazo + interpolación, detección con 1-40 ────────────────
    print("  [2/5] Rechazo + interpolación  (detect=1-40, final=8-30)...")
    m, s, n_disc = _run_rejection_condition(
        fif_paths, channel_names,
        interpolate=True, detection_pipeline=_pipeline_detection_broad(),
        n_total=n_total,
    )
    result["acc_interp_broad_mean"] = m
    result["acc_interp_broad_std"]  = s
    result["eliminados_broad"] = n_disc

    # ── [3/5] Rechazo + interpolación, detección con 8-30 ────────────────
    print("  [3/5] Rechazo + interpolación  (detect=8-30, final=8-30)...")
    m, s, n_disc = _run_rejection_condition(
        fif_paths, channel_names,
        interpolate=True, detection_pipeline=_pipeline_detection_narrow(),
        n_total=n_total,
    )
    result["acc_interp_narrow_mean"] = m
    result["acc_interp_narrow_std"]  = s
    result["eliminados_narrow"] = n_disc

    # ── [4/5] Solo rechazo, detección con 1-40 ───────────────────────────
    print("  [4/5] Solo rechazo, sin interpolación  (detect=1-40, final=8-30)...")
    m, s, _ = _run_rejection_condition(
        fif_paths, channel_names,
        interpolate=False, detection_pipeline=_pipeline_detection_broad(),
        n_total=n_total,
    )
    result["acc_solo_broad_mean"] = m
    result["acc_solo_broad_std"]  = s

    # ── [5/5] Solo rechazo, detección con 8-30 ───────────────────────────
    print("  [5/5] Solo rechazo, sin interpolación  (detect=8-30, final=8-30)...")
    m, s, _ = _run_rejection_condition(
        fif_paths, channel_names,
        interpolate=False, detection_pipeline=_pipeline_detection_narrow(),
        n_total=n_total,
    )
    result["acc_solo_narrow_mean"] = m
    result["acc_solo_narrow_std"]  = s

    return result


# ── Salida ─────────────────────────────────────────────────────────────────────

def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.4f} ±{std:.4f}"


# Claves de accuracy en el orden de filas de la tabla
_COND_KEYS = [
    ("Sin rechazo",        "acc_sin_mean",           "acc_sin_std"),
    ("Interp + rej 1-40", "acc_interp_broad_mean",  "acc_interp_broad_std"),
    ("Interp + rej 8-30", "acc_interp_narrow_mean", "acc_interp_narrow_std"),
    ("rejection 1-40",  "acc_solo_broad_mean",    "acc_solo_broad_std"),
    ("rejection  8-30",  "acc_solo_narrow_mean",   "acc_solo_narrow_std"),
]

_INFO_KEYS = [
    ("Eliminados 1-40", lambda r: str(r["eliminados_broad"])  if r.get("eliminados_broad")  is not None else "—"),
    ("Eliminados 8-30", lambda r: str(r["eliminados_narrow"]) if r.get("eliminados_narrow") is not None else "—"),
    ("Total",           lambda r: str(r.get("total", "—"))),
]


def _print_table(results: list[dict]) -> None:
    subjects = [r["sujeto"] for r in results]

    # Columna 0 = etiqueta de fila, luego una columna por sujeto
    col0_w = max(len(label) for label, *_ in _COND_KEYS + [(k, None) for k, _ in _INFO_KEYS])
    cell_w = max(
        max(len(s) for s in subjects),
        len(_fmt(0.5, 0.05)),  # ancho de una celda de accuracy típica
    )

    def _row(label, cells, best_col=None):
        label_cell = label.ljust(col0_w)
        parts = []
        for i, c in enumerate(cells):
            marker = " *" if i == best_col else "  "
            parts.append(c.ljust(cell_w) + marker)
        return "  " + label_cell + "  |  " + "  |  ".join(parts)

    sep_inner = "-" * (cell_w + 2)
    sep = "  " + "-" * col0_w + "--+--" + "--+--".join(sep_inner for _ in subjects)

    header_cells = [s.ljust(cell_w) + "  " for s in subjects]
    header_label = " " * col0_w
    header_line  = "  " + header_label + "  |  " + "  |  ".join(header_cells)

    print(f"\n\n══ Resultados {N_FOLDS}-Fold CV ══════════════════════════════════════════")
    print(header_line)
    print(sep)

    for label, mean_key, std_key in _COND_KEYS:
        cells = [_fmt(r.get(mean_key), r.get(std_key)) for r in results]
        # mejor accuracy de esta fila no tiene sentido; queremos el mejor por columna (sujeto)
        print(_row(label, cells))

    print(sep)

    # Fila con el mejor % por sujeto (marcada con *)
    print("  " + "mejor".ljust(col0_w) + "  |  " + "  |  ".join(
        _best_cell(results, i, cell_w) for i in range(len(results))
    ))

    print(sep)

    for info_label, fn in _INFO_KEYS:
        cells = [fn(r) for r in results]
        print(_row(info_label, cells))

    print()


def _best_cell(results: list[dict], subj_idx: int, cell_w: int) -> str:
    """Devuelve la etiqueta de condición con mayor accuracy para ese sujeto."""
    r = results[subj_idx]
    best_label, best_val = "—", -1.0
    for label, mean_key, _ in _COND_KEYS:
        v = r.get(mean_key)
        if v is not None and v > best_val:
            best_val, best_label = v, label.strip()
    text = f"{best_label} ({best_val:.4f})" if best_val >= 0 else "—"
    return text.ljust(cell_w) + " *"


def run_comparison() -> list[dict]:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return []

    print(f"Sujetos encontrados: {list(subjects.keys())}")
    results = [_run_subject(name, paths) for name, paths in subjects.items()]
    _print_table(results)
    return results


if __name__ == "__main__":
    run_comparison()
