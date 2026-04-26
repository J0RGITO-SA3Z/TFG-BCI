"""
Compara métricas con y sin congelar el encoder del transformer de MiRepNet.

Por cada sujeto:
  1. Carga todos sus .fif con un único FifDataProvider.
  2. Divide los datos en train (N_TRAIN muestras fijas) y test (el resto).
  3. Evalúa con freeze_encoder=False y freeze_encoder=True.
  4. Muestra una tabla comparativa con accuracy, balanced accuracy, F1 y kappa.
"""
import os
import sys
import glob
import random
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)

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

N_TRAIN         = 40
SEED            = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS     = ["left_hand", "right_hand"]


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


def _split(X: np.ndarray, y: np.ndarray, n_train: int = N_TRAIN) -> tuple:
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


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
    freeze_encoder: bool,
) -> dict:
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
        freeze_encoder=freeze_encoder,
    )

    X_tr, y_tr = epoch_pipeline.process_np(X_train, y_train, shuffle=False)
    X_te, y_te = epoch_pipeline.process_np(X_test,  y_test,  shuffle=False)

    modelo.finetuning(X_tr, y_tr, X_te, y_te, epochs=FINETUNE_EPOCHS)
    preds, _ = modelo.predict_batch(X_te)

    return {
        "acc":      float(accuracy_score(y_te, preds)),
        "bal_acc":  float(balanced_accuracy_score(y_te, preds)),
        "f1":       float(f1_score(y_te, preds, average="macro", zero_division=0)),
        "kappa":    float(cohen_kappa_score(y_te, preds)),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _raw_pipeline() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


# ── Lógica principal ───────────────────────────────────────────────────────────

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    print(f"{'─' * 60}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_raw_pipeline(),
        annotations_names=ANNOTATIONS,
    )
    X_all, y_all, _ = provider.get_data()
    n_total = len(X_all)

    if n_total <= N_TRAIN:
        print(f"  AVISO: solo {n_total} epochs disponibles, se necesitan >{N_TRAIN}. Saltando.")
        return {"sujeto": subj_name, "freeze_off": None, "freeze_on": None, "total": n_total}

    X_tr, y_tr, X_te, y_te = _split(X_all, y_all)

    # ── freeze_encoder = False ─────────────────────────────────────────────
    print("  [1/2] freeze_encoder=False (todos los pesos se actualizan)...")
    metrics_off = _evaluate(X_tr, y_tr, X_te, y_te, channel_names, freeze_encoder=False)
    print(f"    acc={metrics_off['acc']:.4f}  bal_acc={metrics_off['bal_acc']:.4f}"
          f"  f1={metrics_off['f1']:.4f}  kappa={metrics_off['kappa']:.4f}")

    # ── freeze_encoder = True ──────────────────────────────────────────────
    print("  [2/2] freeze_encoder=True  (solo se entrena la cabeza clasificadora)...")
    metrics_on = _evaluate(X_tr, y_tr, X_te, y_te, channel_names, freeze_encoder=True)
    print(f"    acc={metrics_on['acc']:.4f}  bal_acc={metrics_on['bal_acc']:.4f}"
          f"  f1={metrics_on['f1']:.4f}  kappa={metrics_on['kappa']:.4f}")

    return {
        "sujeto":     subj_name,
        "freeze_off": metrics_off,
        "freeze_on":  metrics_on,
        "total":      n_total,
    }


def _fmt(val) -> str:
    return f"{val:.4f}" if val is not None else "—"


def _print_table(results: list[dict]) -> None:
    metrics = ["acc", "bal_acc", "f1", "kappa"]
    headers = ["Sujeto", "Total",
               "ACC (off)", "BalAcc (off)", "F1 (off)", "Kappa (off)",
               "ACC (on)",  "BalAcc (on)",  "F1 (on)",  "Kappa (on)"]

    rows = []
    for r in results:
        off = r.get("freeze_off") or {}
        on  = r.get("freeze_on")  or {}
        row = [
            r["sujeto"],
            str(r.get("total", "—")),
            _fmt(off.get("acc")),     _fmt(off.get("bal_acc")),
            _fmt(off.get("f1")),      _fmt(off.get("kappa")),
            _fmt(on.get("acc")),      _fmt(on.get("bal_acc")),
            _fmt(on.get("f1")),       _fmt(on.get("kappa")),
        ]
        rows.append(row)

    # Fila de medias (ignorando None / "—")
    def _mean_col(col_idx):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[col_idx]))
            except ValueError:
                pass
        return f"{np.mean(vals):.4f}" if vals else "—"

    mean_row = ["MEDIA", ""] + [_mean_col(i) for i in range(2, len(headers))]
    rows.append(mean_row)

    col_w = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def _row(cells):
        return "  " + "  |  ".join(c.ljust(w) for c, w in zip(cells, col_w))

    sep = "  " + "--+--".join("-" * w for w in col_w)

    print("\n\n══ Resultados: freeze_encoder OFF vs ON ══════════════════════════════════")
    print(_row(headers))
    print(sep)
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            print(sep)
        print(_row(row))
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
