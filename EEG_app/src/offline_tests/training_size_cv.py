"""
Evalúa cómo varía el accuracy del modelo según el número exacto de epochs
por clase usados en entrenamiento.

Para cada sujeto y cada valor de N_TRAIN:
  - Se repiten N_FOLDS splits aleatorios: N_TRAIN epochs por clase → train,
    el resto → test.
  - Se promedian los accuracies para reducir la varianza debida a la semilla.

Resultado: gráfica con una línea por sujeto (accuracy vs. epochs de entrenamiento
por clase).
"""
import os
import sys
import glob
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from tqdm import tqdm

SRC_ROOT     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
RECORDINGS   = os.path.abspath(os.path.join(SRC_ROOT, "..", "recordings", "experimento_visual"))
OUTPUT_DIR   = os.path.abspath(os.path.join(SRC_ROOT, "..", "results"))

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

# ── Configuración ───────────────────────────────────────────────────────────────

ANNOTATIONS     = ["left_hand", "right_hand"]
FINETUNE_EPOCHS = 10
N_FOLDS         = 10       # repeticiones por punto para promediar varianza de semilla
# Número exacto de epochs por clase a probar en entrenamiento
N_TRAIN_VALUES  = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def _load_data(fif_paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pipeline = RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])
    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=pipeline,
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()
    channel_names = provider.get_channel_names()
    return X, y, channel_names


def _split_by_class(y: np.ndarray, n_per_class: int, rng: np.random.RandomState
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (train_idx, test_idx) con exactamente n_per_class por clase en train.
    Con n_per_class=0 todo va a test (baseline sin fine-tuning)."""
    if n_per_class == 0:
        return np.array([], dtype=np.intp), np.arange(len(y))
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        rest   = np.setdiff1d(cls_idx, chosen)
        train_idx.append(chosen)
        test_idx.append(rest)
    return np.concatenate(train_idx), np.concatenate(test_idx)


def _evaluate_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    channel_names: list[str],
    seed: int,
) -> float:
    _seed(seed)
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

    X_te, y_te = epoch_pipeline.process_np(X_test, y_test, shuffle=False)

    if len(X_train) > 0:
        X_tr, y_tr = epoch_pipeline.process_np(X_train, y_train, shuffle=False)
        modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)

    preds, _ = modelo.predict_batch(X_te)
    return float(accuracy_score(y_te, preds))


def _evaluate_n_train(
    X: np.ndarray, y: np.ndarray, channel_names: list[str], n_per_class: int,
    pbar: tqdm, subj_name: str,
) -> tuple[float, float]:
    """Repite N_FOLDS splits con distintas semillas y devuelve (mean, std)."""
    accs = []
    for fold in range(N_FOLDS):
        rng = np.random.RandomState(fold * 37 + 13)
        train_idx, test_idx = _split_by_class(y, n_per_class, rng)
        acc = _evaluate_fold(
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            channel_names, seed=fold,
        )
        accs.append(acc)
        pbar.set_postfix({
            "suj":  subj_name,
            "N":    n_per_class,
            "fold": f"{fold + 1}/{N_FOLDS}",
            "acc":  f"{acc:.3f}",
        }, refresh=True)
        pbar.update(1)
    return float(np.mean(accs)), float(np.std(accs))


def _run_subject(
    subj_name: str, fif_paths: list[str], pbar: tqdm,
) -> dict[str, list]:
    tqdm.write(f"\n{'─' * 62}")
    tqdm.write(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    tqdm.write(f"{'─' * 62}")

    X, y, channel_names = _load_data(fif_paths)

    classes, counts = np.unique(y, return_counts=True)
    min_available = int(counts.min())
    tqdm.write(f"  Epochs por clase: {dict(zip(classes, counts))}  (mín={min_available})")

    valid_ns   = [n for n in N_TRAIN_VALUES if n < min_available]
    skipped_ns = [n for n in N_TRAIN_VALUES if n >= min_available]

    if not valid_ns:
        tqdm.write(f"  AVISO: sin epochs suficientes para ningún N_TRAIN. Saltando.")
        pbar.update(len(N_TRAIN_VALUES) * N_FOLDS)
        return {"n_train": [], "mean": [], "std": [], "min_available": min_available}

    # Avanzar la barra por los N que se saltan (sin datos suficientes)
    pbar.update(len(skipped_ns) * N_FOLDS)

    results_n, results_mean, results_std = [], [], []
    for n in valid_ns:
        mean, std = _evaluate_n_train(X, y, channel_names, n, pbar, subj_name)
        results_n.append(n)
        results_mean.append(mean)
        results_std.append(std)
        tqdm.write(f"    N={n:2d}: {mean:.4f} ± {std:.4f}")

    return {
        "n_train":       results_n,
        "mean":          results_mean,
        "std":           results_std,
        "min_available": min_available,
    }


# ── Gráfica ─────────────────────────────────────────────────────────────────────

def _plot(all_results: dict[str, dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for i, (subj, res) in enumerate(all_results.items()):
        if not res["n_train"]:
            continue
        ns    = res["n_train"]
        means = res["mean"]
        stds  = res["std"]
        color = colors[i % len(colors)]
        ax.plot(ns, means, marker="o", label=subj, color=color)
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="azar (50%)")
    ax.set_xlabel("Epochs de entrenamiento por clase", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Accuracy vs. tamaño del conjunto de entrenamiento\n"
        f"({N_FOLDS} folds por punto, media ± desv. típ.)",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "training_size_cv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {out_path}")
    plt.close(fig)


# ── Tabla resumen ───────────────────────────────────────────────────────────────

def _print_summary(all_results: dict[str, dict]) -> None:
    print("\n\n══ Resumen final ══════════════════════════════════════════════")
    for subj, res in all_results.items():
        if not res["n_train"]:
            print(f"  {subj}: sin datos suficientes")
            continue
        best_idx = int(np.argmax(res["mean"]))
        print(
            f"  {subj}:  mejor con N={res['n_train'][best_idx]} epochs/clase"
            f"  →  {res['mean'][best_idx]:.4f} ± {res['std'][best_idx]:.4f}"
            f"  (disponibles/clase: {res['min_available']})"
        )
    print()


# ── Punto de entrada ────────────────────────────────────────────────────────────

def run() -> dict[str, dict]:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return {}

    total_folds = len(subjects) * len(N_TRAIN_VALUES) * N_FOLDS
    print(f"Sujetos encontrados: {list(subjects.keys())}")
    print(f"N_TRAIN a probar: {N_TRAIN_VALUES}  |  folds por punto: {N_FOLDS}")
    print(f"Total de ejecuciones (máx): {total_folds}\n")

    all_results = {}
    with tqdm(
        total=total_folds,
        unit="fold",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds  [{elapsed}<{remaining}  {rate_fmt}]",
        dynamic_ncols=True,
    ) as pbar:
        for name, paths in subjects.items():
            all_results[name] = _run_subject(name, paths, pbar)

    _print_summary(all_results)
    _plot(all_results)
    return all_results


if __name__ == "__main__":
    run()
