"""
Evalúa cómo varía el accuracy en función del número de epochs de entrenamiento
por clase, usando el pipeline CAR + Notch + 8-30 Hz + epoch rejection.

Para cada sujeto y cada N (de N_TRAIN_MIN a N_TRAIN_MAX epochs por clase):
  1. Selecciona aleatoriamente N epochs de cada clase para train.
  2. Usa todos los epochs restantes para test.
  3. Repite N_REPS veces con semillas distintas y promedia el accuracy.

Al final pinta una gráfica con una línea por sujeto.
"""
import os
import sys
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

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
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator
from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

SEED           = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS    = ["left_hand", "right_hand"]
N_TRAIN_MIN    = 5
N_TRAIN_MAX    = 30
N_REPS         = 5   # repeticiones por cada N para promediar


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pipeline_detection() -> RawProcessorPipeline:
    return RawProcessorPipeline([
        NotchFilter(50.0),
        BandpassFilter(1.0, 40.0),
        AnnotationRenamer(LABEL_MAP),
    ])


def _pipeline_final() -> RawProcessorPipeline:
    return RawProcessorPipeline([
        CARReference(),
        NotchFilter(50.0),
        BandpassFilter(8.0, 30.0),
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


def _split_by_class(
    X: np.ndarray, y: np.ndarray, n_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < n_per_class + 1:
            return None, None
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        train_idx.extend(chosen)
        test_idx.extend(np.setdiff1d(idx, chosen))
    return np.array(train_idx), np.array(test_idx)


def _evaluate(X_train, y_train, X_test, y_test, channel_names, seed) -> float:
    _seed_everything(seed)
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

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict | None:
    print(f"\n{'─' * 60}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    print(f"{'─' * 60}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_detection(),
        raw_pipeline_final=_pipeline_final(),
        bad_channel_interpolator=_bad_channel_interpolator(channel_names),
        interpolate_bad_channels=True,
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        print(f"  AVISO: menos de 2 clases. Saltando.")
        return None

    min_count = int(counts.min())
    max_n = min(N_TRAIN_MAX, min_count - 1)
    if max_n < N_TRAIN_MIN:
        print(f"  AVISO: solo {min_count} epochs por clase, mínimo requerido {N_TRAIN_MIN + 1}. Saltando.")
        return None

    train_sizes = list(range(N_TRAIN_MIN, max_n + 1))
    mean_accs = []

    for n in train_sizes:
        rep_accs = []
        for rep in range(N_REPS):
            seed = SEED + rep * 100
            train_idx, test_idx = _split_by_class(X, y, n, seed)
            if train_idx is None:
                break
            acc = _evaluate(X[train_idx], y[train_idx], X[test_idx], y[test_idx], channel_names, seed)
            rep_accs.append(acc)
        mean_acc = float(np.mean(rep_accs)) if rep_accs else None
        mean_accs.append(mean_acc)
        print(f"  n={n:2d}/clase → acc={mean_acc:.4f}  (media de {len(rep_accs)} reps, test={len(test_idx)} epochs)")

    return {"sujeto": subj_name, "train_sizes": train_sizes, "accs": mean_accs}


def _plot_results(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, r in enumerate(results):
        color = cmap(i % 10)
        valid = [(n, a) for n, a in zip(r["train_sizes"], r["accs"]) if a is not None]
        if not valid:
            continue
        ns, accs = zip(*valid)
        ax.plot(ns, accs, marker="o", markersize=4, linewidth=1.8, color=color, label=r["sujeto"])

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Azar (50%)")
    ax.set_xlabel("Epochs de entrenamiento por clase", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. tamaño del conjunto de entrenamiento\n(pipeline: CAR + Notch + 8-30 Hz)", fontsize=13)
    ax.set_xlim(N_TRAIN_MIN - 0.5, N_TRAIN_MAX + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(N_TRAIN_MIN, N_TRAIN_MAX + 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_size_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nGráfica guardada en: {out_path}")
    plt.show()


def run_comparison() -> None:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return

    print(f"Sujetos encontrados: {list(subjects.keys())}")
    results = [r for name, paths in subjects.items() if (r := _run_subject(name, paths)) is not None]

    if results:
        _plot_results(results)


if __name__ == "__main__":
    run_comparison()
