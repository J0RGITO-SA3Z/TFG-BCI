"""
Curva de overfitting: accuracy en train y test vs. número de épocas de
fine-tuning, con split fijo 30 trials/clase para train y 30 para test.

Solo se procesan sujetos con ≈60 epochs por clase (120 en total).

Para cada fold se hace UN solo entrenamiento de MAX_EPOCHS épocas y se
extrae la curva completa del history que devuelve finetuning(). Esto
evita re-entrenar desde cero para cada punto del eje X.
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

# ── Configuración ────────────────────────────────────────────────────────────────

ANNOTATIONS       = ["left_hand", "right_hand"]
N_TRAIN_PER_CLASS = 30    # trials por clase en entrenamiento (fijo)
N_TEST_PER_CLASS  = 30    # trials por clase en test (fijo)
N_FOLDS           = 10    # repeticiones para reducir varianza de semilla
MAX_EPOCHS        = 25    # épocas de fine-tuning (eje X: 0..MAX_EPOCHS)
TARGET_PER_CLASS  = 60
CLASS_TOLERANCE   = 5     # acepta [55, 65] epochs por clase


# ── Helpers ──────────────────────────────────────────────────────────────────────

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
        raise FileNotFoundError(f"No se encontró: {RECORDINGS}")
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
    return X, y, provider.get_channel_names()


def _has_target_samples(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    lo, hi = TARGET_PER_CLASS - CLASS_TOLERANCE, TARGET_PER_CLASS + CLASS_TOLERANCE
    return bool(np.all((counts >= lo) & (counts <= hi)))


def _split_fixed(y: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """30 train / 30 test por clase, selección aleatoria."""
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen  = rng.choice(cls_idx, size=N_TRAIN_PER_CLASS, replace=False)
        rest    = np.setdiff1d(cls_idx, chosen)
        test    = rng.choice(rest, size=N_TEST_PER_CLASS, replace=False)
        train_idx.append(chosen)
        test_idx.append(test)
    return np.concatenate(train_idx), np.concatenate(test_idx)


def _run_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    channel_names: list[str],
    seed: int,
) -> tuple[list[float], list[float]]:
    """
    Un solo entrenamiento de MAX_EPOCHS épocas.
    Devuelve (train_accs, val_accs) con MAX_EPOCHS+1 valores cada una:
      índice 0 → antes de entrenar, índices 1..MAX_EPOCHS → tras cada época.
    train_acc y val_acc vienen en % desde finetuning(), se devuelven en [0,1].
    """
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])
    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=ANNOTATIONS,
    )

    X_tr_p, y_tr_p = ep_pipeline.process_np(X_tr, y_tr, shuffle=False)
    X_te_p, y_te_p = ep_pipeline.process_np(X_te, y_te, shuffle=False)

    # Punto 0: modelo sin fine-tuning
    from sklearn.metrics import accuracy_score
    tr0_preds, _ = modelo.predict_batch(X_tr_p)
    te0_preds, _ = modelo.predict_batch(X_te_p)
    tr0 = accuracy_score(y_tr_p, tr0_preds)
    te0 = accuracy_score(y_te_p, te0_preds)

    # Un solo entrenamiento → history con MAX_EPOCHS entradas
    history = modelo.finetuning(X_tr_p, y_tr_p, X_te_p, y_te_p, epochs=MAX_EPOCHS)

    # train_acc y val_acc vienen en % (ver comentario en MiRepNetInterface)
    train_accs = [tr0] + [h["train_acc"] / 100.0 for h in history]
    val_accs   = [te0] + [h["val_acc"]   / 100.0 for h in history]

    return train_accs, val_accs


def _run_subject(
    subj_name: str, fif_paths: list[str], pbar: tqdm,
) -> dict | None:
    tqdm.write(f"\n{'─' * 62}")
    tqdm.write(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    tqdm.write(f"{'─' * 62}")

    X, y, channel_names = _load_data(fif_paths)
    classes, counts = np.unique(y, return_counts=True)
    tqdm.write(f"  Epochs por clase: {dict(zip(classes, counts))}")

    if not _has_target_samples(y):
        tqdm.write(f"  → No tiene {TARGET_PER_CLASS}±{CLASS_TOLERANCE} epochs/clase. Saltando.")
        pbar.update(N_FOLDS)
        return None

    if int(counts.min()) < N_TRAIN_PER_CLASS + N_TEST_PER_CLASS:
        tqdm.write(f"  → Insuficientes muestras para split {N_TRAIN_PER_CLASS}/{N_TEST_PER_CLASS}. Saltando.")
        pbar.update(N_FOLDS)
        return None

    # shape: (N_FOLDS, MAX_EPOCHS+1)
    all_train = np.zeros((N_FOLDS, MAX_EPOCHS + 1))
    all_val   = np.zeros((N_FOLDS, MAX_EPOCHS + 1))

    for fold in range(N_FOLDS):
        rng = np.random.RandomState(fold * 37 + 13)
        train_idx, test_idx = _split_fixed(y, rng)
        tr_curve, te_curve = _run_fold(
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            channel_names, seed=fold,
        )
        all_train[fold] = tr_curve
        all_val[fold]   = te_curve

        final_gap = tr_curve[-1] - te_curve[-1]
        pbar.set_postfix({
            "suj":  subj_name,
            "fold": f"{fold + 1}/{N_FOLDS}",
            "tr":   f"{tr_curve[-1]:.3f}",
            "te":   f"{te_curve[-1]:.3f}",
            "gap":  f"{final_gap:+.3f}",
        }, refresh=True)
        pbar.update(1)
        tqdm.write(
            f"    fold {fold + 1}/{N_FOLDS}:  "
            f"train@{MAX_EPOCHS}={tr_curve[-1]:.4f}  "
            f"test@{MAX_EPOCHS}={te_curve[-1]:.4f}  "
            f"gap={final_gap:+.4f}"
        )

    return {
        "epochs":      list(range(MAX_EPOCHS + 1)),
        "train_mean":  all_train.mean(axis=0).tolist(),
        "train_std":   all_train.std(axis=0).tolist(),
        "test_mean":   all_val.mean(axis=0).tolist(),
        "test_std":    all_val.std(axis=0).tolist(),
    }


# ── Gráfica ──────────────────────────────────────────────────────────────────────

def _plot(all_results: dict[str, dict]) -> None:
    valid = {k: v for k, v in all_results.items() if v is not None}
    if not valid:
        print("No hay resultados que graficar.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = plt.cm.tab10.colors
    n_subj = len(valid)

    fig, axes = plt.subplots(n_subj, 1, figsize=(10, 4 * n_subj), squeeze=False)

    for row, (subj, res) in enumerate(valid.items()):
        ax  = axes[row, 0]
        eps = res["epochs"]
        tr_m, tr_s = np.array(res["train_mean"]), np.array(res["train_std"])
        te_m, te_s = np.array(res["test_mean"]),  np.array(res["test_std"])

        ax.plot(eps, tr_m, color=colors[0], label="Train accuracy")
        ax.fill_between(eps, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color=colors[0])

        ax.plot(eps, te_m, color=colors[1], label="Test accuracy")
        ax.fill_between(eps, te_m - te_s, te_m + te_s, alpha=0.15, color=colors[1])

        ax.fill_between(eps, te_m, tr_m, alpha=0.10, color="red", label="Brecha (overfitting)")

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Azar (50%)")
        ax.set_title(subj, fontsize=12)
        ax.set_xlabel("Épocas de fine-tuning")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, MAX_EPOCHS)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Overfitting: train vs. test accuracy vs. épocas de fine-tuning\n"
        f"(split {N_TRAIN_PER_CLASS}/{N_TEST_PER_CLASS} por clase, {N_FOLDS} folds, media ± desv. típ.)",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "overfitting_cv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en: {out_path}")
    plt.close(fig)


# ── Tabla resumen ─────────────────────────────────────────────────────────────────

def _print_summary(all_results: dict[str, dict]) -> None:
    print("\n\n══ Resumen final (overfitting por épocas) ══════════════════════")
    print(f"  Split: {N_TRAIN_PER_CLASS} train / {N_TEST_PER_CLASS} test por clase  |  {N_FOLDS} folds\n")
    for subj, res in all_results.items():
        if res is None:
            print(f"  {subj}: descartado")
            continue
        tr_m = res["train_mean"]
        te_m = res["test_mean"]
        gaps = [t - e for t, e in zip(tr_m, te_m)]
        max_g_ep  = res["epochs"][int(np.argmax(gaps))]
        best_te_ep = res["epochs"][int(np.argmax(te_m))]
        print(
            f"  {subj}:  "
            f"max_gap={max(gaps):+.4f} en época {max_g_ep}  |  "
            f"mejor_test={max(te_m):.4f} en época {best_te_ep}"
        )
    print()


# ── Punto de entrada ──────────────────────────────────────────────────────────────

def run() -> dict[str, dict]:
    subjects = _discover_subjects()
    if not subjects:
        print(f"No se encontraron sujetos en {RECORDINGS}")
        return {}

    total_folds = len(subjects) * N_FOLDS
    print(f"Sujetos encontrados: {list(subjects.keys())}")
    print(f"Filtrando sujetos con {TARGET_PER_CLASS}±{CLASS_TOLERANCE} epochs/clase")
    print(f"Split fijo: {N_TRAIN_PER_CLASS} train / {N_TEST_PER_CLASS} test por clase")
    print(f"Épocas: {MAX_EPOCHS}  |  Folds: {N_FOLDS}  |  Total entrenamientos (máx): {total_folds}\n")

    all_results: dict[str, dict | None] = {}
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
