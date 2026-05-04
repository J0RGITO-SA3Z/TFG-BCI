"""
Compara el accuracy de distintas combinaciones de filtros de preprocesado.

Por cada sujeto y combinación de filtros:
  1. Carga todos sus .fif con FifDataProvider.
  2. Aplica el pipeline: CAR -> Notch -> [frecuencia opcional] -> ICA
  3. Para epoch rejection usa siempre BandpassFilter(1, 40) en detección.
  4. Divide en train (N_TRAIN fijos) y test (el resto).
  5. Evalúa con MiRepNet finetuneado.

La tabla final muestra las combinaciones de filtros en filas y los sujetos en columnas.
"""
import os
import sys
import glob
import random
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

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
from components.RawProcessing.CARReference import CARReference
from components.RawProcessing.NotchFilter import NotchFilter
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.ICAProcessor import ICAProcessor
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator
from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

K_FOLDS         = 5
SEED            = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS     = ["left_hand", "right_hand"]

# ── Combinaciones de filtros a comparar ────────────────────────────────────────
# Orden fijo: CAR -> Notch(50) -> [banda opcional] -> ICA
# Cada entrada: (etiqueta, car, notch, bandpass, l_freq, h_freq, ica)
# Agrupadas en 3 secciones según el filtro de frecuencia.
FILTER_CONFIGS = [
    # ── Sección 1: Sin filtro de paso de banda ─────────────────────────────
    ("CAR + Notch + ICA",           True,  True,  False, None, None, True),
    ("CAR + Notch",                 True,  True,  False, None, None, False),
    ("CAR + ICA",                   True,  False, False, None, None, True),
    ("Notch + ICA",                 False, True,  False, None, None, True),
    ("CAR",                    True,  False, False, None, None, False),
    ("Notch",                  False, True,  False, None, None, False),
    ("ICA",                    False, False, False, None, None, True),
    ("Sin preprocesado",            False, False, False, None, None, False),
    # ── Sección 2: Paso banda 1–40 Hz ──────────────────────────────────────
    ("CAR + Notch + 1-40 + ICA",    True,  True,  True,  1.0, 40.0, True),
    ("CAR + Notch + 1-40",          True,  True,  True,  1.0, 40.0, False),
    ("CAR + 1-40 + ICA",            True,  False, True,  1.0, 40.0, True),
    ("Notch + 1-40 + ICA",          False, True,  True,  1.0, 40.0, True),
    ("CAR + 1-40",                  True,  False, True,  1.0, 40.0, False),
    ("Notch + 1-40",                False, True,  True,  1.0, 40.0, False),
    ("1-40 + ICA",             False, False, True,  1.0, 40.0, True),
    ("1-40",                   False, False, True,  1.0, 40.0, False),
    # ── Sección 3: Paso banda 8–30 Hz ──────────────────────────────────────
    ("CAR + Notch + 8-30 + ICA",    True,  True,  True,  8.0, 30.0, True),
    ("CAR + Notch + 8-30",          True,  True,  True,  8.0, 30.0, False),
    ("CAR + 8-30 + ICA",            True,  False, True,  8.0, 30.0, True),
    ("Notch + 8-30 + ICA",          False, True,  True,  8.0, 30.0, True),
    ("CAR + 8-30",                  True,  False, True,  8.0, 30.0, False),
    ("Notch + 8-30",                False, True,  True,  8.0, 30.0, False),
    ("8-30 + ICA",             False, False, True,  8.0, 30.0, True),
    ("8-30",                   False, False, True,  8.0, 30.0, False),
]

SECTION_BREAKS = {"CAR + Notch + ICA", "CAR + Notch + 1-40 + ICA", "CAR + Notch + 8-30 + ICA"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_subject_files(subject_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(subject_dir, "*_raw.fif")))


def _discover_subjects() -> dict[str, list[str]]:
    subjects = {}
    if not os.path.isdir(RECORDINGS):
        raise FileNotFoundError(f"No encontrado: {RECORDINGS}")
    for entry in sorted(os.listdir(RECORDINGS)):
        subject_dir = os.path.join(RECORDINGS, entry)
        if os.path.isdir(subject_dir) and entry.startswith("suj"):
            files = _get_subject_files(subject_dir)
            if files:
                subjects[entry] = files
    return subjects


def _balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.RandomState(SEED)
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min_count, replace=False)
        indices.append(chosen)
    indices = np.sort(np.concatenate(indices))
    return X[indices], y[indices], len(X) - len(indices)


def _cross_validate(X: np.ndarray, y: np.ndarray, channel_names: list[str]) -> tuple[float, dict]:
    X, y, n_dropped = _balance_classes(X, y)
    classes = np.unique(y)
    lh_label, rh_label = classes[0], classes[1]

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs, rh_trains, lh_trains, rh_tests, lh_tests = [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        y_tr, y_te = y[train_idx], y[test_idx]
        rh_trains.append(int(np.sum(y_tr == rh_label)))
        lh_trains.append(int(np.sum(y_tr == lh_label)))
        rh_tests.append(int(np.sum(y_te == rh_label)))
        lh_tests.append(int(np.sum(y_te == lh_label)))
        acc = _evaluate(X[train_idx], y_tr, X[test_idx], y_te, channel_names)
        print(f"      fold {fold}/{K_FOLDS}: {acc:.4f}")
        fold_accs.append(acc)
    stats = {
        "n_dropped_balance": n_dropped,
        "rh_train": int(round(np.mean(rh_trains))),
        "lh_train": int(round(np.mean(lh_trains))),
        "rh_test":  int(round(np.mean(rh_tests))),
        "lh_test":  int(round(np.mean(lh_tests))),
    }
    return float(np.mean(fold_accs)), stats


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_pipeline(car: bool, notch: bool, bandpass: bool, l_freq, h_freq, ica: bool) -> RawProcessorPipeline:
    steps = []
    if car:
        steps.append(CARReference())
    if notch:
        steps.append(NotchFilter(50.0))
    if bandpass:
        steps.append(BandpassFilter(l_freq, h_freq))
    if ica:
        steps.append(ICAProcessor())
    steps.append(AnnotationRenamer(LABEL_MAP))
    return RawProcessorPipeline(steps)


def _pipeline_detection() -> RawProcessorPipeline:
    """Pipeline fijo para epoch rejection (detección): siempre 1-40 Hz."""
    return RawProcessorPipeline([
        NotchFilter(50.0),
        BandpassFilter(1.0, 40.0),
        AnnotationRenamer(LABEL_MAP),
    ])


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


def _evaluate(X_train, y_train, X_test, y_test, channel_names) -> float:
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


# ── Lógica principal ───────────────────────────────────────────────────────────

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    """Evalúa todas las combinaciones de filtros para un sujeto."""
    print(f"\n{'─' * 60}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    print(f"{'─' * 60}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    # Epoch rejection count (misma lógica que epoch_rejection_comparison.py)
    provider_plain = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=RawProcessorPipeline([BandpassFilter(1.0, 40.0), AnnotationRenamer(LABEL_MAP)]),
        annotations_names=ANNOTATIONS,
    )
    X_plain, _, _ = provider_plain.get_data()
    n_total_raw = len(X_plain)

    provider_rej = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_detection(),
        raw_pipeline_final=RawProcessorPipeline([BandpassFilter(1.0, 40.0), AnnotationRenamer(LABEL_MAP)]),
        bad_channel_interpolator=_bad_channel_interpolator(channel_names),
        interpolate_bad_channels=True,
        annotations_names=ANNOTATIONS,
    )
    X_rej, _, _ = provider_rej.get_data()
    n_discarded = n_total_raw - len(X_rej)
    print(f"  Epoch rejection: {n_discarded}/{n_total_raw} eliminados")

    results = {"sujeto": subj_name, "n_discarded": n_discarded}
    _stats_set = False

    for label, car, notch, bandpass, l_freq, h_freq, ica in FILTER_CONFIGS:
        print(f"  [{label}]...")

        final_pipeline = _build_pipeline(car, notch, bandpass, l_freq, h_freq, ica)
        interp = _bad_channel_interpolator(channel_names)

        provider = FifDataProvider(
            fif_paths=fif_paths,
            raw_pipeline_detection=_pipeline_detection(),
            raw_pipeline_final=final_pipeline,
            bad_channel_interpolator=interp,
            interpolate_bad_channels=True,
            annotations_names=ANNOTATIONS,
        )

        X, y, _ = provider.get_data()

        clases, counts = np.unique(y, return_counts=True)
        if len(clases) < 2 or counts.min() < K_FOLDS:
            print(f"    AVISO: epochs insuficientes por clase {dict(zip(clases, counts))}. Saltando.")
            results[label] = None
            continue

        acc, fold_stats = _cross_validate(X, y, channel_names)
        results[label] = acc
        print(f"    Accuracy media ({K_FOLDS}-fold CV): {acc:.4f}  (epochs: {len(X)})")

        if not _stats_set:
            results["rh_train"] = fold_stats["rh_train"]
            results["lh_train"] = fold_stats["lh_train"]
            results["rh_test"]  = fold_stats["rh_test"]
            results["lh_test"]  = fold_stats["lh_test"]
            _stats_set = True

    return results


def _print_table(results: list[dict], subject_names: list[str]) -> None:
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    labels = [cfg[0] for cfg in FILTER_CONFIGS]

    # Mejor accuracy por sujeto (columna)
    best_per_subj = {}
    for subj in subject_names:
        vals = [r.get(lb) for r in results for lb in labels if r["sujeto"] == subj and r.get(lb) is not None]
        best_per_subj[subj] = max(vals) if vals else None

    # Media por combinación (fila) y mejor combinación global
    means = {}
    for lb in labels:
        vals = [r.get(lb) for r in results if r.get(lb) is not None]
        means[lb] = float(np.mean(vals)) if vals else None
    best_label = max((lb for lb in labels if means[lb] is not None), key=lambda lb: means[lb])

    col_w_label = max(len(lb) for lb in labels)
    col_w_subj  = max(max(len(s) for s in subject_names), 8)
    col_w_mean  = max(len("Media"), 8)

    def _fmt(val, highlight=False):
        s = f"{val:.4f}" if val is not None else "  —   "
        return f"{BOLD}{s}{RESET}" if highlight else s

    all_cols = subject_names + ["Media"]
    col_widths = [col_w_subj] * len(subject_names) + [col_w_mean]

    header_cells = ["Combinación de filtros".ljust(col_w_label)] + [s.center(w) for s, w in zip(all_cols, col_widths)]
    sep = "-+-".join(["-" * col_w_label] + ["-" * w for w in col_widths])

    print("\n\n══ Resultados: accuracy por filtro y sujeto ══════════════════════")
    print("  " + "  |  ".join(header_cells))
    print("  " + sep)

    for label in labels:
        if label in SECTION_BREAKS:
            print("  " + sep)

        is_best_row = (label == best_label)
        row_label = f"{BOLD}{label.ljust(col_w_label)}{RESET}" if is_best_row else label.ljust(col_w_label)
        row = [row_label]

        for subj, w in zip(subject_names, col_widths):
            r = next(res for res in results if res["sujeto"] == subj)
            val = r.get(label)
            highlight = (val is not None and val == best_per_subj[subj])
            row.append(_fmt(val, highlight).center(w))

        mean_val = means[label]
        row.append(_fmt(mean_val, is_best_row).center(col_w_mean))

        print("  " + "  |  ".join(row))

    # ── Filas de estadísticas por sujeto ──────────────────────────────────────
    stat_rows = [
        ("Ep. eliminados",    "n_discarded"),
        ("Derecha (train)",   "rh_train"),
        ("Izquierda (train)", "lh_train"),
        ("Derecha (test)",    "rh_test"),
        ("Izquierda (test)",  "lh_test"),
    ]
    print("  " + sep)
    for row_name, key in stat_rows:
        row = [row_name.ljust(col_w_label)]
        for subj, w in zip(subject_names, col_widths):
            r = next(res for res in results if res["sujeto"] == subj)
            val = r.get(key)
            row.append((str(val) if val is not None else "—").center(w))
        row.append("".center(col_w_mean))
        print("  " + "  |  ".join(row))

    print()


def run_comparison() -> None:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return

    print(f"Sujetos encontrados: {list(subjects.keys())}")
    results = [_run_subject(name, paths) for name, paths in subjects.items()]
    _print_table(results, list(subjects.keys()))


if __name__ == "__main__":
    run_comparison()
