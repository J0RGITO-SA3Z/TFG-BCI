"""
run_moabb_experiment.py
=======================
Entrena MiRepNet sobre un dataset de MOABB para un sujeto concreto
y visualiza los resultados con PerformanceViewer.

Cambia las variables de la sección CONFIG para probar distintos datasets.
"""
import sys
import os
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Imports visualización ─────────────────────────────────────────────────────
from utils.Performance_Viewer import PerformanceViewer

# ── Imports modelInterface ──────────────────────────────────────────────────────────
from model_interface.MiRepNetInterface import MiRepNetInterface

# ── MOABB ─────────────────────────────────────────────────────────────────────
import moabb
moabb.set_log_level("ERROR")

# ── DataProviders ─────────────────────────────────────────────────────────────────────
from DataProvider.MoabbDataProvider import MoabbDataProvider
from DataProvider.FifDataProvider import FifDataProvider

# ── Data Processing ─────────────────────────────────────────────────────────────────────
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.EpochNormalizer import EpochNormalizer
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.ClassEventRemover import ClassEventRemover

# =============================================================================
# CONFIG — cambia aquí para probar distintos datasets/sujetos
# =============================================================================
DATASET_NAME = "BNCI2014001"   # "BNCI2014001" | "BNCI2014004" | "BNCI2015001"
SUBJECT_IDX  = 0               # índice del sujeto (0-based)
EPOCHS       = 20
BATCH_SIZE   = 32
LR           = 1e-3
VAL_SPLIT    = 0.2
SEED         = 42
# =============================================================================
# Pipeline de procesamiento sobre epochs (se ejecuta después de epoquizar)

def EA_Matrix(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    refEA : numpy array
        reference matrix for Euclidean Alignment of shape (num_channels, num_channels)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)

    return refEA

def run_moabb_piepline(dataset, subjectIdx,epochs, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    dataProvider = MoabbDataProvider(dataset,subjectIdx)
    X, y, classes = dataProvider.get_data()

    epoch_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),        # interpola/reordena canales a la topología objetivo 
    ])

    num_clases = len(classes)

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    X_train, y_train = epoch_pipeline.process_np(X_train, y_train,shuffle=False)
    X_val, y_val = epoch_pipeline.process_np(X_val, y_val,shuffle=False)

    final_val_acc = modelo.finetuning_processed(X_train, y_train, epochs=epochs)
    preds_array, probs_array = modelo.predict_batch_preprocessed(X_val)

    viewer = PerformanceViewer()
    viewer.plot_downstream2(probs_array, y_val, class_names = classes)

def prueba_clasificarRes_sinEntrenar(fif_names, epochs, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand", "feet", "rest"])
    X, y, classes = dataProvider.get_data()

    epoch_training_pipeline = EpochProcessorPipeline([
        ClassEventRemover(classes_to_remove = classes.index("rest") if "rest" in classes else []), # eliminar clase "rest" si existe
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    epoch_validation_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    num_clases = len(classes)

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    X_train, y_train = epoch_training_pipeline.process_np(X_train, y_train,shuffle=False)
    X_val, y_val = epoch_validation_pipeline.process_np(X_val, y_val,shuffle=False)

    historico = modelo.finetuning_processed(X_train, y_train, epochs=epochs)
    preds_array, probs_array = modelo.predict_batch_preprocessed(X_val)

    viewer = PerformanceViewer()
    viewer.summary(historico)
    viewer.plot_downstream2(probs_array, y_val, class_names = classes)

def run_fif_piepline(fif_names, epochs, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand", "feet"])
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    epoch_training_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    epoch_validation_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    X_train, y_train = epoch_training_pipeline.process_np(X_train, y_train,shuffle=False)
    X_val, y_val = epoch_validation_pipeline.process_np(X_val, y_val,shuffle=False)

    historico = modelo.finetuning_processed(X_train, y_train, epochs=epochs)
    preds_array, probs_array = modelo.predict_batch_preprocessed(X_val)

    viewer = PerformanceViewer()
    viewer.summary(historico)
    viewer.plot_downstream2(probs_array, y_val, class_names = classes)

    
def run_fif_piepline2(train_fif_names,val_fif_names, epochs, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    train_dataProvider = FifDataProvider(fif_paths = train_fif_names, annotations_names=["left_hand", "right_hand", "feet"])
    validation_dataProvider = FifDataProvider(fif_paths = val_fif_names, annotations_names=["left_hand", "right_hand", "feet"])

    X_train, y_train, classes = train_dataProvider.get_data()
    X_val, y_val, _ = validation_dataProvider.get_data()

    num_clases = len(classes)

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = train_dataProvider.get_channel_names())

    EA_Matrix_train = EA_Matrix(X_train)
    EA_Matrix_val = EA_Matrix(X_val)

    # ── Comparación de matrices EA ────────────────────────────────────
    diff = EA_Matrix_train - EA_Matrix_val
    frobenius_norm = np.linalg.norm(diff, 'fro')
    relative_diff = frobenius_norm / np.linalg.norm(EA_Matrix_train, 'fro')
    print(f"EA diff  — Frobenius norm: {frobenius_norm:.6f}")
    print(f"EA diff  — Relative diff:  {relative_diff:.4%}")
    print(f"EA diff  — Max abs diff:   {np.max(np.abs(diff)):.6f}")
    print(f"EA diff  — Mean abs diff:  {np.mean(np.abs(diff)):.6f}")

    epoch_training_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(EA_Matrix_train),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = train_dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    epoch_validation_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(EA_Matrix_train),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = validation_dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    X_train, y_train = epoch_training_pipeline.process_np(X_train, y_train,shuffle=False)
    X_val, y_val = epoch_validation_pipeline.process_np(X_val, y_val,shuffle=False)

    historico = modelo.finetuning_processed(X_train, y_train, epochs=epochs)
    preds_array, probs_array = modelo.predict_batch_preprocessed(X_val)

    viewer = PerformanceViewer()
    viewer.summary(historico)
    viewer.plot_downstream2(probs_array, y_val, class_names = classes)



if __name__ == "__main__":
    input_type = input("¿Cargar datos de MOABB[1], de archivos .fif[2]?: ").strip().lower()
    if input_type == "1":
        run_moabb_piepline(
            dataset=    "BNCI2014001", # "BNCI2014001" | "BNCI2014004" | "BNCI2015001"
            epochs       = 10,
            val_split    = 0.4,
            seed         = SEED,
        )
    elif input_type == "2":

        fif_names = ["EEG_controller_app/recordings/suj2_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_4_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_5_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_6_raw.fif"]

        run_fif_piepline(
            fif_names    = fif_names,
            epochs       = 10,
            val_split    = 0.4,
            seed         = SEED,
        )

    elif input_type == "3":
        fif_names = ["EEG_controller_app/recordings/suj2_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_4_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_5_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_6_raw.fif"]

        prueba_clasificarRes_sinEntrenar(
            fif_names    = fif_names,
            epochs       = 10,
            val_split    = 0.4,
            seed         = SEED,
        )
    elif input_type == "4":
        train_fif_names = [ "EEG_controller_app/recordings/suj2_3_raw.fif"]
        val_fif_names = ["EEG_controller_app/recordings/suj2_4_raw.fif"]

        run_fif_piepline2(
            train_fif_names    = train_fif_names,
            val_fif_names      = val_fif_names,
            epochs             = 10,
            val_split          = 0.4,
            seed               = SEED,
        )