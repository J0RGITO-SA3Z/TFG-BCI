"""
Compara el accuracy con cuatro métodos de interpolación de canales para cada sujeto.

Condiciones evaluadas (4 en total):
  1. SpatialInterpolator         — IDW con posiciones 2D de MiRepNet
  2. MNEPositionIDWInterpolator  — IDW con posiciones 3D de MNE (standard_1005)
  3. SphericalSplineInterpolator — interpolación spline esférico de MNE
  4. ZeroInterpolator            — canales faltantes puestos a 0 (baseline)

Todas las condiciones usan: sin epoch rejection, filtro 8-30 Hz, K-Fold CV estratificado.
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
from components.EpochProcessing.MNEPositionIDWInterpolator import MNEPositionIDWInterpolator
from components.EpochProcessing.SphericalSplineInterpolator import SphericalSplineInterpolator
from components.EpochProcessing.ZeroInterpolator import ZeroInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

N_FOLDS         = 5
TRAIN_SIZE      = 0.15
SEED            = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS     = ["left_hand", "right_hand"]
MIN_PER_CLASS   = 4


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


_INTERPOLATORS = {
    "spatial":  lambda ch: SpatialInterpolator(actual_channel_positions=ch),
    "mne_idw":  lambda ch: MNEPositionIDWInterpolator(actual_channel_positions=ch),
    "spline":   lambda ch: SphericalSplineInterpolator(actual_channel_positions=ch),
    "zero":     lambda ch: ZeroInterpolator(actual_channel_positions=ch),
}


def _evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    channel_names: list[str],
    interp_key: str,
) -> float:
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interpolator = _INTERPOLATORS[interp_key](channel_names)

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        interpolator,
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
    interp_key: str,
) -> tuple[float, float]:
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, train_size=TRAIN_SIZE, random_state=SEED)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        acc = _evaluate(
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            channel_names, interp_key,
        )
        accs.append(acc)
        print(f"      Fold {fold + 1}/{N_FOLDS}: acc={acc:.4f}")

    return float(np.mean(accs)), float(np.std(accs))


def _has_enough_samples(y: np.ndarray) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    return len(classes) == len(ANNOTATIONS) and all(c >= N_FOLDS * MIN_PER_CLASS for c in counts)


def _raw_pipeline() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


# ── Lógica principal ───────────────────────────────────────────────────────────

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    print(f"\n{'─' * 62}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)  —  {N_FOLDS}-Fold CV estratificado")
    print(f"{'─' * 62}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_raw_pipeline(),
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()
    n_total = len(X)

    result = {
        "sujeto":           subj_name,
        "acc_spatial_mean": None, "acc_spatial_std": None,
        "acc_mne_idw_mean": None, "acc_mne_idw_std": None,
        "acc_spline_mean":  None, "acc_spline_std":  None,
        "acc_zero_mean":    None, "acc_zero_std":    None,
        "total":            n_total,
    }

    if not _has_enough_samples(y):
        print(f"  AVISO: {n_total} epochs insuficientes para {N_FOLDS} folds. Saltando sujeto.")
        return result

    # ── [1/4] SpatialInterpolator ─────────────────────────────────────────
    print("  [1/4] SpatialInterpolator (IDW posiciones MiRepNet)...")
    mean, std = _kfold_evaluate(X, y, channel_names, interp_key="spatial")
    result["acc_spatial_mean"] = mean
    result["acc_spatial_std"]  = std
    print(f"    → {mean:.4f} ± {std:.4f}")

    # ── [2/4] MNEPositionIDWInterpolator ──────────────────────────────────
    print("  [2/4] MNEPositionIDWInterpolator (IDW posiciones MNE)...")
    mean, std = _kfold_evaluate(X, y, channel_names, interp_key="mne_idw")
    result["acc_mne_idw_mean"] = mean
    result["acc_mne_idw_std"]  = std
    print(f"    → {mean:.4f} ± {std:.4f}")

    # ── [3/4] SphericalSplineInterpolator ─────────────────────────────────
    print("  [3/4] SphericalSplineInterpolator (MNE spline)...")
    mean, std = _kfold_evaluate(X, y, channel_names, interp_key="spline")
    result["acc_spline_mean"] = mean
    result["acc_spline_std"]  = std
    print(f"    → {mean:.4f} ± {std:.4f}")

    # ── [4/4] ZeroInterpolator ────────────────────────────────────────────
    print("  [4/4] ZeroInterpolator (baseline ceros)...")
    mean, std = _kfold_evaluate(X, y, channel_names, interp_key="zero")
    result["acc_zero_mean"] = mean
    result["acc_zero_std"]  = std
    print(f"    → {mean:.4f} ± {std:.4f}")

    return result


# ── Salida ─────────────────────────────────────────────────────────────────────

def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.4f} ±{std:.4f}"


_COND_KEYS = [
    ("IDW MiRepNet",          "acc_spatial_mean", "acc_spatial_std"),
    ("IDW MNE pos",           "acc_mne_idw_mean", "acc_mne_idw_std"),
    ("SphericalSpline (MNE)", "acc_spline_mean",  "acc_spline_std"),
    ("Zero (baseline)",       "acc_zero_mean",    "acc_zero_std"),
]

_INFO_KEYS = [
    ("Total", lambda r: str(r.get("total", "—"))),
]


def _print_table(results: list[dict]) -> None:
    subjects = [r["sujeto"] for r in results]

    col0_w = max(len(label) for label, *_ in _COND_KEYS + [(k, None) for k, _ in _INFO_KEYS])
    cell_w = max(
        max(len(s) for s in subjects),
        len(_fmt(0.5, 0.05)),
    )

    def _row(label, cells):
        label_cell = label.ljust(col0_w)
        parts = [c.ljust(cell_w) for c in cells]
        return "  " + label_cell + "  |  " + "  |  ".join(parts)

    sep_inner = "-" * (cell_w + 2)
    sep = "  " + "-" * col0_w + "--+--" + "--+--".join(sep_inner for _ in subjects)

    header_cells = [s.ljust(cell_w) for s in subjects]
    header_line  = "  " + " " * col0_w + "  |  " + "  |  ".join(header_cells)

    print(f"\n\n══ Resultados {N_FOLDS}-Fold CV  ─  IDW-MiRepNet vs IDW-MNE vs Spline vs Zero ══")
    print(header_line)
    print(sep)

    for label, mean_key, std_key in _COND_KEYS:
        cells = [_fmt(r.get(mean_key), r.get(std_key)) for r in results]
        print(_row(label, cells))

    print(sep)

    for info_label, fn in _INFO_KEYS:
        cells = [fn(r) for r in results]
        print(_row(info_label, cells))

    print(sep)
    for label, keys in [
        ("Δ (IDW MiRepNet − Zero)", ("acc_spatial_mean", "acc_zero_mean")),
        ("Δ (IDW MNE − Zero)",      ("acc_mne_idw_mean", "acc_zero_mean")),
        ("Δ (Spline − Zero)",       ("acc_spline_mean",  "acc_zero_mean")),
    ]:
        diffs = []
        for r in results:
            a, b = r.get(keys[0]), r.get(keys[1])
            diffs.append(f"{a - b:+.4f}" if (a is not None and b is not None) else "—")
        print(_row(label, diffs))
    print()


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
