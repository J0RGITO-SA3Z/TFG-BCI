"""
Evalúa cómo varía el rendimiento de MiRepNet en función del número de muestras
de fine-tuning usando Validación Cruzada de Monte Carlo (MCCV).

Por cada valor de n_train en N_TRAIN_VALUES:
  1. Se generan N_REPEATS particiones aleatorias estratificadas (train / test).
  2. Se hace fine-tuning desde cero (pesos preentrenados) en cada partición.
  3. Se agregan acc, bal_acc, f1 y kappa como media ± desv.std.
  4. Se comparan freeze_encoder=False y freeze_encoder=True.

Configuración clave (cambia aquí para ajustar el experimento):
  SUBJECT_NAME   - sujeto a analizar
  N_TRAIN_VALUES - valores de n_train a explorar
  N_REPEATS      - repeticiones MCCV por cada n_train
  FREEZE_MODES   - modos de congelación del encoder a comparar
  FINETUNE_EPOCHS- épocas de entrenamiento por partición

Tiempo estimado: ~N_TRAIN_VALUES × N_REPEATS × FREEZE_MODES × FINETUNE_EPOCHS ×
                  (tiempo por epoch).  Reduce N_REPEATS o usa valores seleccionados
                  en N_TRAIN_VALUES para acortar la ejecución.
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

# ── Configuración ──────────────────────────────────────────────────────────────
SUBJECT_NAME    = "suj5"
ANNOTATIONS     = ["left_hand", "right_hand"]
SEED            = 42
FINETUNE_EPOCHS = 10
N_REPEATS       = 5
N_TRAIN_VALUES  = list(range(1, 51))   # 1 … 50 muestras de fine-tuning
FREEZE_MODES    = [False, True]        # False = todos los pesos, True = solo cabeza
MIN_TEST        = 10                   # mínimo de epochs requeridos en el conjunto de test


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _raw_pipeline() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


def _load_subject(subject_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subject_dir = os.path.join(RECORDINGS, subject_name)
    fif_paths = sorted(glob.glob(os.path.join(subject_dir, "*_raw.fif")))
    if not fif_paths:
        raise FileNotFoundError(f"No se encontraron archivos .fif en {subject_dir}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()
    provider = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_raw_pipeline(),
        annotations_names=ANNOTATIONS,
    )
    X, y, _ = provider.get_data()
    return X, y, channel_names


def _make_splits(y: np.ndarray, n_train: int, n_repeats: int) -> list[tuple]:
    """
    Devuelve n_repeats pares (train_idx, test_idx) con exactamente n_train
    muestras en entrenamiento. Usa muestreo estratificado cuando n_train es
    suficiente para garantizar al menos una muestra por clase; en caso contrario
    usa muestreo aleatorio simple.
    """
    n_classes = len(np.unique(y))
    if n_train >= n_classes:
        sss = StratifiedShuffleSplit(
            n_splits=n_repeats, train_size=n_train, random_state=SEED
        )
        return [(tr, te) for tr, te in sss.split(np.zeros(len(y)), y)]

    # n_train < n_classes: muestreo aleatorio con semillas distintas
    rng = np.random.RandomState(SEED)
    splits = []
    for _ in range(n_repeats):
        idx = rng.permutation(len(y))
        splits.append((idx[:n_train], idx[n_train:]))
    return splits


def _new_epoch_pipeline(channel_names: list[str]) -> EpochProcessorPipeline:
    """
    Crea siempre una instancia nueva porque EuclideanAlignment es con estado:
    almacena la matriz de referencia calculada en la primera llamada.
    """
    return EpochProcessorPipeline([
        EuclideanAlignment(),
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])


def _evaluate_single(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    channel_names: list[str],
    freeze_encoder: bool,
) -> dict | None:
    """
    Entrena y evalúa una única partición.
    Devuelve None si el conjunto de test no tiene suficientes muestras o clases.
    """
    if len(X_test) < MIN_TEST:
        return None
    if len(np.unique(y_test)) < 2:
        return None

    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pipelines independientes: EuclideanAlignment no debe compartir estado
    # entre train y test (calcula la matriz de referencia sobre train y la
    # reutiliza en test, tal como ocurriría en producción).
    ea_train = EuclideanAlignment()
    ea_test_placeholder = EuclideanAlignment()  # se sobreescribe con la matriz de train

    pipeline_train = EpochProcessorPipeline([
        ea_train,
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])
    X_tr, y_tr = pipeline_train.process_np(X_train, y_train, shuffle=False)

    # Reutilizar la matriz de referencia calculada sobre train en el test
    ea_test_placeholder.matrix = ea_train.matrix
    pipeline_test = EpochProcessorPipeline([
        ea_test_placeholder,
        SpatialInterpolator(actual_channel_positions=channel_names),
    ])
    X_te, y_te = pipeline_test.process_np(X_test, y_test, shuffle=False)

    modelo = MiRepNetInterface(
        device=device,
        weight_path=WEIGHT_PATH,
        training_clases=ANNOTATIONS,
        freeze_encoder=freeze_encoder,
    )
    modelo.finetuning(X_tr, y_tr, epochs=FINETUNE_EPOCHS)
    preds, _ = modelo.predict_batch(X_te)

    return {
        "acc":     float(accuracy_score(y_te, preds)),
        "bal_acc": float(balanced_accuracy_score(y_te, preds)),
        "f1":      float(f1_score(y_te, preds, average="macro", zero_division=0)),
        "kappa":   float(cohen_kappa_score(y_te, preds)),
    }


def _aggregate(metrics: list[dict]) -> dict:
    result = {}
    for key in ["acc", "bal_acc", "f1", "kappa"]:
        vals = [m[key] for m in metrics]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"]  = float(np.std(vals))
    return result


# ── Lógica principal ───────────────────────────────────────────────────────────

def _run_sweep(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
) -> dict[bool, list[dict]]:
    """
    Para cada (freeze_mode, n_train) ejecuta N_REPEATS evaluaciones MCCV.
    Devuelve {freeze_mode: [{n_train, n_valid, metrics...}, ...]}.
    """
    n_total = len(X)
    results: dict[bool, list[dict]] = {mode: [] for mode in FREEZE_MODES}

    for n_train in N_TRAIN_VALUES:
        if n_train + MIN_TEST > n_total:
            print(f"  n_train={n_train:>3}: saltando "
                  f"(total={n_total}, mínimo test={MIN_TEST})")
            for mode in FREEZE_MODES:
                results[mode].append({"n_train": n_train, "skipped": True})
            continue

        splits = _make_splits(y, n_train, N_REPEATS)

        for freeze in FREEZE_MODES:
            label = "on " if freeze else "off"
            fold_metrics = []

            for rep_idx, (tr_idx, te_idx) in enumerate(splits):
                print(
                    f"  n_train={n_train:>3} | freeze={label} | "
                    f"rep {rep_idx + 1}/{N_REPEATS} ...",
                    end=" ", flush=True,
                )
                m = _evaluate_single(
                    X[tr_idx], y[tr_idx],
                    X[te_idx], y[te_idx],
                    channel_names,
                    freeze_encoder=freeze,
                )
                if m is not None:
                    print(f"acc={m['acc']:.3f}")
                    fold_metrics.append(m)
                else:
                    print("ignorado (test insuficiente o mono-clase)")

            if fold_metrics:
                agg = _aggregate(fold_metrics)
            else:
                agg = {f"{k}_{s}": float("nan")
                       for k in ["acc", "bal_acc", "f1", "kappa"]
                       for s in ["mean", "std"]}

            results[freeze].append({
                "n_train": n_train,
                "n_valid": len(fold_metrics),
                **agg,
            })

    return results


# ── Salida ─────────────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "    —    "
    return f"{mean:.3f}±{std:.3f}"


def _print_results(results: dict[bool, list[dict]]) -> None:
    modes = [(False, "freeze=OFF (todos los pesos)"),
             (True,  "freeze=ON  (solo cabeza clasificadora)")]

    print("\n\n══ Resultados MCCV: n_train vs métricas (media ± desv.std) ══════════════")

    for freeze, label in modes:
        rows = results[freeze]
        print(f"\n── {label} ────────────────────────────────────────────────────────")
        hdr = (f"{'n_train':>8}  {'valid':>5}  "
               f"{'ACC':^13}  {'BalACC':^13}  {'F1':^13}  {'Kappa':^13}")
        print(hdr)
        print("─" * len(hdr))
        for r in rows:
            if r.get("skipped"):
                print(f"{r['n_train']:>8}  {'—':>5}  {'saltado'}")
                continue
            print(
                f"{r['n_train']:>8}  {r['n_valid']:>5}  "
                f"{_fmt(r['acc_mean'],     r['acc_std']):^13}  "
                f"{_fmt(r['bal_acc_mean'], r['bal_acc_std']):^13}  "
                f"{_fmt(r['f1_mean'],      r['f1_std']):^13}  "
                f"{_fmt(r['kappa_mean'],   r['kappa_std']):^13}"
            )
    print()


def run_sweep() -> dict[bool, list[dict]]:
    subject_dir = os.path.join(RECORDINGS, SUBJECT_NAME)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Directorio de sujeto no encontrado: {subject_dir}")

    print(f"Cargando datos de {SUBJECT_NAME}...")
    X, y, channel_names = _load_subject(SUBJECT_NAME)
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Total epochs : {len(X)}")
    print(f"  Clases       : { {c: int(n) for c, n in zip(classes, counts)} }")
    print(f"  n_train range: {N_TRAIN_VALUES[0]}…{N_TRAIN_VALUES[-1]} "
          f"({len(N_TRAIN_VALUES)} valores)")
    print(f"  Repeticiones : {N_REPEATS} MCCV por punto")
    print(f"  Freeze modes : {FREEZE_MODES}")
    print(f"  Épocas FT    : {FINETUNE_EPOCHS}\n")

    results = _run_sweep(X, y, channel_names)
    _print_results(results)
    return results


if __name__ == "__main__":
    run_sweep()
