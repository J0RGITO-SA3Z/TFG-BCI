"""
Compara el accuracy con y sin epoch rejection para cada sujeto.

Por cada sujeto:
  1. Carga todos sus .fif juntos con un único FifDataProvider.
  2. Divide los datos en train (N_TRAIN muestras fijas) y test (el resto).
  3. Evalúa con el pipeline sin rechazo y con el pipeline con rechazo.
  4. Muestra una tabla comparativa con accuracy y epochs eliminados.
"""
import os
import sys
import glob
import random
import numpy as np
import torch

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
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator
from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface

N_TRAIN = 40
SEED    = 42
FINETUNE_EPOCHS = 10
ANNOTATIONS = ["left_hand", "right_hand"]


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
) -> float:
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


# ── Pipelines ──────────────────────────────────────────────────────────────────

def _pipeline_no_rejection() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(8, 30.0), AnnotationRenamer(LABEL_MAP)])


def _pipeline_detection() -> RawProcessorPipeline:
    return RawProcessorPipeline([BandpassFilter(1, 40.0), AnnotationRenamer(LABEL_MAP)])


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

def _run_subject(subj_name: str, fif_paths: list[str]) -> dict:
    print(f"\n{'─' * 55}")
    print(f"  {subj_name}  ({len(fif_paths)} sesiones)")
    print(f"{'─' * 55}")

    channel_names = FifDataProvider(fif_paths=fif_paths).get_channel_names()

    # ── Sin rechazo ────────────────────────────────────────────────────────
    print("  [1/2] Sin epoch rejection...")
    provider_plain = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_no_rejection(),
        annotations_names=ANNOTATIONS,
    )
    X_all, y_all, _ = provider_plain.get_data()
    n_total = len(X_all)

    if n_total <= N_TRAIN:
        print(f"  AVISO: solo {n_total} epochs disponibles, se necesitan >{N_TRAIN}. Saltando.")
        return {"sujeto": subj_name, "acc_sin": None, "acc_con": None, "eliminados": None}

    X_tr, y_tr, X_te, y_te = _split(X_all, y_all)
    acc_plain = _evaluate(X_tr, y_tr, X_te, y_te, channel_names)
    print(f"    Accuracy sin rechazo: {acc_plain:.4f}")

    # ── Con rechazo + interpolación ────────────────────────────────────────
    print("  [2/3] Con epoch rejection + interpolación de canales malos...")
    interp = _bad_channel_interpolator(channel_names)
    provider_rej = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_detection(),
        raw_pipeline_final=_pipeline_final(),
        bad_channel_interpolator=interp,
        interpolate_bad_channels=True,
        annotations_names=ANNOTATIONS,
    )
    X_rej, y_rej, _ = provider_rej.get_data()
    n_discarded = n_total - len(X_rej)

    acc_rej = None
    if len(X_rej) <= N_TRAIN:
        print(f"  AVISO: tras rechazo solo quedan {len(X_rej)} epochs. Saltando evaluación.")
    else:
        X_tr_r, y_tr_r, X_te_r, y_te_r = _split(X_rej, y_rej)
        acc_rej = _evaluate(X_tr_r, y_tr_r, X_te_r, y_te_r, channel_names)
        print(f"    Accuracy con rechazo + interp: {acc_rej:.4f}  (eliminados: {n_discarded}/{n_total})")

    # ── Solo rechazo, sin interpolación ───────────────────────────────────
    print("  [3/3] Con epoch rejection, sin interpolación de canales malos...")
    interp2 = _bad_channel_interpolator(channel_names)
    provider_rej_only = FifDataProvider(
        fif_paths=fif_paths,
        raw_pipeline_detection=_pipeline_detection(),
        raw_pipeline_final=_pipeline_final(),
        bad_channel_interpolator=interp2,
        interpolate_bad_channels=False,
        annotations_names=ANNOTATIONS,
    )
    X_rej2, y_rej2, _ = provider_rej_only.get_data()

    acc_rej_only = None
    if len(X_rej2) <= N_TRAIN:
        print(f"  AVISO: tras rechazo solo quedan {len(X_rej2)} epochs. Saltando evaluación.")
    else:
        X_tr_r2, y_tr_r2, X_te_r2, y_te_r2 = _split(X_rej2, y_rej2)
        acc_rej_only = _evaluate(X_tr_r2, y_tr_r2, X_te_r2, y_te_r2, channel_names)
        print(f"    Accuracy solo rechazo:         {acc_rej_only:.4f}  (eliminados: {n_discarded}/{n_total})")

    return {
        "sujeto":        subj_name,
        "acc_sin":       acc_plain,
        "acc_con_interp": acc_rej,
        "acc_con_solo":  acc_rej_only,
        "eliminados":    n_discarded,
        "total":         n_total,
    }


def _print_table(results: list[dict]) -> None:
    headers = ["Sujeto", "Sin rechazo", "Rechazo+interp", "Solo rechazo", "Eliminados", "Total"]
    rows = []
    for r in results:
        acc_sin  = f"{r['acc_sin']:.4f}"        if r.get("acc_sin")        is not None else "—"
        acc_interp = f"{r['acc_con_interp']:.4f}" if r.get("acc_con_interp") is not None else "—"
        acc_only = f"{r['acc_con_solo']:.4f}"   if r.get("acc_con_solo")   is not None else "—"
        elim     = str(r["eliminados"])          if r.get("eliminados")     is not None else "—"
        total    = str(r.get("total", "—"))
        rows.append([r["sujeto"], acc_sin, acc_interp, acc_only, elim, total])

    col_w = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def _row(cells):
        return "  " + "  |  ".join(c.ljust(w) for c, w in zip(cells, col_w))

    sep = "  " + "--+--".join("-" * w for w in col_w)

    print("\n\n══ Resultados comparativos ════════════════════════════════════════")
    print(_row(headers))
    print(sep)
    for row in rows:
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
