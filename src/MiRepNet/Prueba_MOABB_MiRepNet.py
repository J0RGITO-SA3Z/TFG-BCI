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

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Imports visualización ─────────────────────────────────────────────────────
from utils.Performance_Viewer import PerformanceViewer

# ── Imports modelInterface ──────────────────────────────────────────────────────────
from model_interface.modeloGuarro import ModeloGuarro

# ── MOABB ─────────────────────────────────────────────────────────────────────
import moabb
moabb.set_log_level("ERROR")

# ── DataProviders ─────────────────────────────────────────────────────────────────────
from DataProvider.MoabbDataProvider import MoabbDataProvider
from DataProvider.FifDataProvider import FifDataProvider

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

def run_moab(dataset_name, subject_idx, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

   
    # 1. Datos
    dataProvider = MoabbDataProvider(dataset_name=dataset_name, subject_idx=subject_idx)
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)

    print(f"Clases: {classes}\n")

    modelo = ModeloGuarro(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())
    history,_,_ = modelo.experimento(X, y, val_split=val_split, batch_size=batch_size, seed=seed, epochs=epochs)
    viewer = PerformanceViewer()
    viewer.plot_downstream(y_val, probs_Array, class_names = classes)

    return history

def run_fif(fif_names, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
   
    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand", "feet"])
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)
    print(f"Clases: {classes}\n")

    modelo = ModeloGuarro(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())
    history,_,_ = modelo.experimento(X, y, val_split=val_split, batch_size=batch_size, seed=seed, epochs=epochs)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    
    acc, probs_Array,pred_array = modelo.validate(X_val, y_val)
    

    print(f"Val acc: {acc:.1f}%")

    return history

def run_fif_separado(fif_names, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
   
    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand","feet",  "rest"])
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)
    print(f"Clases: {classes}\n")

    modelo = ModeloGuarro(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    history = modelo.finetuning(X_train, y_train, epochs=20, seed=seed, batch_size=batch_size)

    acc, probs_Array,pred_array = modelo.validate(X_val, y_val)
    print(f"Val acc: {acc:.1f}%")

    print(f"Predicciones: {pred_array}\n")
    print(f"Reales: {y_val}\n")
    print(probs_Array)
    print(probs_Array.shape)

    viewer = PerformanceViewer()
    viewer.summary(history)
    viewer.plot_downstream2(probs_Array, y_val, class_names = classes)

    return history


if __name__ == "__main__":
    input_type = input("¿Cargar datos de MOABB[1], de archivos .fif[2] o de archivos .fif con pipeline separado[3]?: ").strip().lower()
    if input_type == "1":
        run_moab(
            dataset_name = DATASET_NAME,
            subject_idx  = SUBJECT_IDX,
            epochs       = EPOCHS,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = VAL_SPLIT,
            seed         = SEED,
        )
    elif input_type == "2":

        fif_names = ["EEG_controller_app/recordings/suj2_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_4_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_5_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_6_raw.fif"]

        run_fif(
            fif_names    = fif_names,
            epochs       = EPOCHS,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = 0.2,
            seed         = SEED,
        )

    elif input_type == "3":

        fif_names = ["EEG_controller_app/recordings/suj2_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_4_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_5_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj2_6_raw.fif"]

        run_fif_separado(
            fif_names    = fif_names,
            epochs       = 10,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = 0.2,
            seed         = SEED,
        )