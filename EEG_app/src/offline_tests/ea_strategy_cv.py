"""
Compara dos estrategias de Euclidean Alignment en CV estratificado para todos los sujetos.

Condición A — EA independiente:
    Se calcula una matriz EA separada para los datos de finetuning y otra
    para los datos de test. Cada conjunto se alinea con sus propias estadísticas.

Condición B — EA compartida:
    La matriz EA se calcula únicamente con los datos de finetuning y se aplica
    también al conjunto de test. Simula el uso real donde no hay datos de test
    disponibles en el momento de calcular la referencia.

Pipeline de preprocesado: BandpassFilter(8, 30) + AnnotationRenamer.
Validación: StratifiedShuffleSplit con TRAIN_SIZE fracción para finetuning.
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
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

import mne

N_FOLDS         = 5
TRAIN_SIZE      = 0.15
SEED            = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS     = ["left_hand", "right_hand"]
MIN_PER_CLASS   = 4


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_subject_files(subject_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(subject_dir, "*_raw.fif")))


def _discover_subjects() -> dict[str, list[str]]:
    subjects = {}
    if not os.path.isdir(RECORDINGS):
        raise FileNotFoundError(f"No se encontró: {RECORDINGS}")
    for entry in sorted(os.listdir(RECORDINGS)):
        subject_dir = os.path.join(RECORDINGS, entry)
        if os.path.isdir(subject_dir) and entry.startswith("suj"):
            files = _get_subject_files(subject_dir)
            if files:
                subjects[entry] = files
    return subjects


def _build_raw_pipeline() -> RawProcessorPipeline:
    return RawProcessorPipeline([
        BandpassFilter(8.0, 30.0),
        AnnotationRenamer(LABEL_MAP),
    ])


def _has_enough_samples(y: np.ndarray) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    return len(classes) == len(ANNOTATIONS) and all(c >= N_FOLDS * MIN_PER_CLASS for c in counts)


def _balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SEED)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    keep = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        keep.append(rng.choice(idx, size=min_count, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


# ── Evaluación ─────────────────────────────────────────────────────────────────

def _evaluate_independent(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    channel_names: list[str],
) -> float:
    """EA independiente: cada conjunto calcula su propia matriz de referencia."""
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline_train = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])
    pipeline_test = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])

    X_tr, y_tr = pipeline_train.process_np(X_train, y_train, shuffle=False)
    X_te, y_te = pipeline_test.process_np(X_test,  y_test,  shuffle=False)

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=ANNOTATIONS,
    )
    modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)
    preds, _ = modelo.predict_batch(X_te)
    return float(accuracy_score(y_te, preds))


def _evaluate_shared(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    channel_names: list[str],
) -> float:
    """EA compartida: la matriz se calcula en training y se reutiliza en test."""
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ea = EuclideanAlignment()
    spatial = SpatialInterpolator(actual_channel_positions=channel_names)

    pipeline = EpochProcessorPipeline([ea, spatial])

    # Primera llamada: calcula y fija la matriz EA a partir de train
    X_tr, y_tr = pipeline.process_np(X_train, y_train, shuffle=False)
    # Segunda llamada: reutiliza la misma matriz (ea.matrix ya está guardado)
    X_te, y_te = pipeline.process_np(X_test, y_test, shuffle=False)

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=ANNOTATIONS,
    )
    modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)
    preds, _ = modelo.predict_batch(X_te)
    return float(accuracy_score(y_te, preds))


# ── K-Fold CV ──────────────────────────────────────────────────────────────────

def _kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
    condition: str,  # "independent" | "shared"
) -> tuple[float, float]:
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, train_size=TRAIN_SIZE, random_state=SEED)
    evaluate_fn = _evaluate_independent if condition == "independent" else _evaluate_shared
    accs = []
    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
        acc = evaluate_fn(X[train_idx], y[train_idx], X[test_idx], y[test_idx], channel_names)
        accs.append(acc)
        print(f"      fold {fold}/{N_FOLDS}: {acc:.4f}")
    return float(np.mean(accs)), float(np.std(accs))


# ── Por sujeto ─────────────────────────────────────────────────────────────────

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    print(f"\n{'─' * 62}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)  —  {N_FOLDS}-Fold CV estratificado")
    print(f"{'─' * 62}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_build_raw_pipeline(),
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()

    result = {
        "sujeto": subj_name,
        "total": len(X),
        "ind_mean": None, "ind_std": None,
        "shr_mean": None, "shr_std": None,
    }

    if not _has_enough_samples(y):
        print(f"  AVISO: {len(X)} epochs insuficientes para {N_FOLDS} folds. Saltando.")
        return result

    X, y = _balance_classes(X, y)
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Balance: {dict(zip(classes, counts))}  (total={len(y)})")

    print("  [A] EA independiente...")
    m, s = _kfold_cv(X, y, channel_names, condition="independent")
    result["ind_mean"], result["ind_std"] = m, s
    print(f"    → {m:.4f} ± {s:.4f}")

    print("  [B] EA compartida (train → test)...")
    m, s = _kfold_cv(X, y, channel_names, condition="shared")
    result["shr_mean"], result["shr_std"] = m, s
    print(f"    → {m:.4f} ± {s:.4f}")

    return result


# ── Tabla de resultados ────────────────────────────────────────────────────────

def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.4f} ±{std:.4f}"


def _print_table(results: list[dict]) -> None:
    subjects = [r["sujeto"] for r in results]
    cell_w = max(len(_fmt(0.5, 0.05)), max(len(s) for s in subjects))
    col0_w = len("EA compartida")

    def _sep():
        return "  " + "-" * col0_w + "--+--" + "--+--".join("-" * (cell_w + 2) for _ in subjects)

    def _row(label, cells):
        return "  " + label.ljust(col0_w) + "  |  " + "  |  ".join(
            c.ljust(cell_w) + "  " for c in cells
        )

    header_cells = [s.ljust(cell_w) + "  " for s in subjects]
    print(f"\n\n══ Resultados {N_FOLDS}-Fold CV ══════════════════════════════════════════")
    print("  " + " " * col0_w + "  |  " + "  |  ".join(header_cells))
    print(_sep())
    print(_row("EA independiente", [_fmt(r["ind_mean"], r["ind_std"]) for r in results]))
    print(_row("EA compartida",    [_fmt(r["shr_mean"], r["shr_std"]) for r in results]))
    print(_sep())
    print(_row("Total epochs",     [str(r["total"]) for r in results]))
    print()


# ── Entrada ────────────────────────────────────────────────────────────────────

def main() -> None:
    mne.set_log_level("WARNING")
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return

    print(f"Sujetos encontrados: {list(subjects.keys())}")
    results = [_run_subject(name, paths) for name, paths in subjects.items()]
    _print_table(results)


if __name__ == "__main__":
    main()
