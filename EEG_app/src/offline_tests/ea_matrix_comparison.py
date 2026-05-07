"""
Compara las matrices de alineamiento euclideo entre grabaciones de suj1 y suj2.

Por cada grabación:
  1. Carga el .fif con un pipeline BandpassFilter(8, 30) + AnnotationRenamer.
  2. Calcula la matriz de referencia EA (media de covarianzas por trial).

Luego construye una tabla simétrica (grabaciones x grabaciones) donde cada
celda es la diferencia relativa elemento a elemento promediada:
    d(A, B) = mean_ij( |A_ij - B_ij| / ((|A_ij| + |B_ij|) / 2) )
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import mne

SRC_ROOT     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, "..", ".."))
RECORDINGS   = os.path.abspath(os.path.join(SRC_ROOT, "..", "recordings", "experimento_visual"))

for _p in [PROJECT_ROOT, SRC_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.DataProvider.FifDataProvider import FifDataProvider, LABEL_MAP
from components.RawProcessing.RawProcessorPipeline import RawProcessorPipeline
from components.RawProcessing.BandpassFilter import BandpassFilter
from components.RawProcessing.AnnotationRenamer import AnnotationRenamer
from components.EpochProcessing.EuclideanAlignment import Calculate_EA_Matrix

SUBJECTS    = ["suj1", "suj3"]
ANNOTATIONS = ["left_hand", "right_hand", "feet"]
L_FREQ, H_FREQ = 8.0, 30.0


def _build_pipeline() -> RawProcessorPipeline:
    return RawProcessorPipeline([
        BandpassFilter(L_FREQ, H_FREQ),
        AnnotationRenamer(LABEL_MAP),
    ])


def _get_recordings(subject: str) -> list[str]:
    subject_dir = os.path.join(RECORDINGS, subject)
    return sorted(glob.glob(os.path.join(subject_dir, "*_raw.fif")))


def _load_ea_matrix(fif_path: str) -> np.ndarray:
    provider = FifDataProvider(
        fif_paths=fif_path,
        raw_pipeline_detection=_build_pipeline(),
    )
    X, y, _ = provider.get_data()
    # X shape: (n_epochs, n_channels, n_times)
    return Calculate_EA_Matrix(X)


def _elementwise_relative_diff(A: np.ndarray, B: np.ndarray) -> float:
    """mean_ij( |A_ij - B_ij| / ((|A_ij| + |B_ij|) / 2) )"""
    denom = (np.abs(A) + np.abs(B)) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denom == 0, 0.0, np.abs(A - B) / denom)
    return float(np.mean(ratio))


def main():
    # Recoge todos los .fif de los sujetos seleccionados
    recordings: list[tuple[str, str]] = []  # (label, fif_path)
    for suj in SUBJECTS:
        files = _get_recordings(suj)
        if not files:
            print(f"[AVISO] No se encontraron grabaciones para {suj} en {RECORDINGS}")
            continue
        for f in files:
            label = f"{suj}_{os.path.basename(f).replace('_raw.fif', '').split('_')[-1]}"
            recordings.append((label, f))

    if not recordings:
        raise RuntimeError("No se encontraron grabaciones.")

    labels = [r[0] for r in recordings]
    n = len(recordings)

    print(f"Grabaciones encontradas ({n}):")
    for lbl, path in recordings:
        print(f"  {lbl}: {path}")

    # Calcula matrices EA para cada grabación
    print("\nCalculando matrices EA...")
    matrices: list[np.ndarray] = []
    for lbl, path in recordings:
        print(f"  Cargando {lbl}...")
        mat = _load_ea_matrix(path)
        matrices.append(mat)
        print(f"    -> matriz shape: {mat.shape}")

    # Construye tabla de comparación
    table = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                table[i, j] = 0.0
            else:
                table[i, j] = _elementwise_relative_diff(matrices[i], matrices[j])

    df = pd.DataFrame(table, index=labels, columns=labels)

    print("\n=== Norma de Frobenius relativa entre matrices EA ===")
    print("(d(A,B) = mean_ij(|A_ij-B_ij| / ((|A_ij|+|B_ij|)/2)) — simétrica)")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(df.to_string())

    # Guarda CSV en la misma carpeta
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ea_matrix_comparison.csv")
    df.to_csv(out_path)
    print(f"\nTabla guardada en: {out_path}")


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()
