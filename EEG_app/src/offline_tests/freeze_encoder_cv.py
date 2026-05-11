"""
Compara el accuracy de MiRepNet con freeze_encoder=True vs freeze_encoder=False.

Para cada sujeto se realiza una validación cruzada de K_FOLDS pliegues
estratificada, con FINETUNE_EPOCHS epochs de fine-tuning en cada pliegue.
Clases: left_hand y right_hand.

Resultado: tabla de resultados por consola y gráfica de barras comparativa.
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
from sklearn.model_selection import StratifiedKFold
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
K_FOLDS         = 5
CONDITIONS      = [False, True]   # freeze_encoder: False → fine-tuning completo, True → solo cabeza


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


def _evaluate_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    channel_names: list[str],
    freeze: bool,
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
        freeze_encoder=freeze,
    )

    X_tr, y_tr = epoch_pipeline.process_np(X_train, y_train, shuffle=False)
    X_te, y_te = epoch_pipeline.process_np(X_test,  y_test,  shuffle=False)

    modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)

    preds, _ = modelo.predict_batch(X_te)
    return float(accuracy_score(y_te, preds))


def _run_condition(
    X: np.ndarray, y: np.ndarray, channel_names: list[str],
    freeze: bool, pbar: tqdm, subj_name: str,
) -> tuple[float, float]:
    """K-fold estratificado para una condición; devuelve (mean, std)."""
    skf  = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    accs = []
    label = "freeze=True " if freeze else "freeze=False"

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        acc = _evaluate_fold(
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            channel_names,
            freeze=freeze,
            seed=fold_idx,
        )
        accs.append(acc)
        pbar.set_postfix({
            "suj":   subj_name,
            "cond":  label,
            "fold":  f"{fold_idx + 1}/{K_FOLDS}",
            "acc":   f"{acc:.3f}",
        }, refresh=True)
        pbar.update(1)

    return float(np.mean(accs)), float(np.std(accs))


def _run_subject(
    subj_name: str, fif_paths: list[str], pbar: tqdm,
) -> dict:
    tqdm.write(f"\n{'─' * 62}")
    tqdm.write(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    tqdm.write(f"{'─' * 62}")

    X, y, channel_names = _load_data(fif_paths)

    classes, counts = np.unique(y, return_counts=True)
    tqdm.write(f"  Epochs por clase: {dict(zip(classes, counts))}")

    results = {}
    for freeze in CONDITIONS:
        mean, std = _run_condition(X, y, channel_names, freeze, pbar, subj_name)
        label = "freeze" if freeze else "no_freeze"
        results[label] = {"mean": mean, "std": std}
        tqdm.write(f"    {label:10s}: {mean:.4f} ± {std:.4f}")

    return results


# ── Gráfica ─────────────────────────────────────────────────────────────────────

def _plot(all_results: dict[str, dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = list(all_results.keys())
    x        = np.arange(len(subjects))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.6), 6))

    for i, (label, color) in enumerate([("no_freeze", "#4C72B0"), ("freeze", "#DD8452")]):
        means = [all_results[s][label]["mean"] for s in subjects]
        stds  = [all_results[s][label]["std"]  for s in subjects]
        bars  = ax.bar(x + (i - 0.5) * width, means, width,
                       yerr=stds, capsize=4,
                       label=label, color=color, alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="azar (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"freeze_encoder: True vs False\n"
        f"({K_FOLDS} folds, {FINETUNE_EPOCHS} epochs fine-tuning, media ± desv. típ.)",
        fontsize=13,
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "freeze_encoder_cv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {out_path}")
    plt.close(fig)


# ── Tabla resumen ───────────────────────────────────────────────────────────────

def _print_summary(all_results: dict[str, dict]) -> None:
    print("\n\n══ Resumen final ══════════════════════════════════════════════")
    header = f"  {'Sujeto':<12}  {'no_freeze':>18}  {'freeze':>18}  {'Δ (freeze−nof.)':>18}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for subj, res in all_results.items():
        nf   = res["no_freeze"]
        fr   = res["freeze"]
        diff = fr["mean"] - nf["mean"]
        print(
            f"  {subj:<12}  "
            f"{nf['mean']:.4f} ± {nf['std']:.4f}  "
            f"{fr['mean']:.4f} ± {fr['std']:.4f}  "
            f"{diff:+.4f}"
        )
    print()


# ── Punto de entrada ────────────────────────────────────────────────────────────

def run() -> dict[str, dict]:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return {}

    total_folds = len(subjects) * len(CONDITIONS) * K_FOLDS
    print(f"Sujetos encontrados: {list(subjects.keys())}")
    print(f"Condiciones: freeze_encoder ∈ {CONDITIONS}  |  folds: {K_FOLDS}  |  epochs: {FINETUNE_EPOCHS}")
    print(f"Total de ejecuciones: {total_folds}\n")

    all_results: dict[str, dict] = {}
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
