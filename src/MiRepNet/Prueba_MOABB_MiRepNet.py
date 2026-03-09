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
from model_interface.modeloGuarro import MiRepNetInterface

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
from epoch_processing.EuclideanAlignmentNotCentred import EuclideanAlignmentNotCentred

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

def run_moab(dataset_name, subject_idx, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

   
    # 1. Datos
    dataProvider = MoabbDataProvider(dataset_name=dataset_name, subject_idx=subject_idx)
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)

    print(f"Clases: {classes}\n")

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())
    history, epoch_predictions, epoch_probabilities = modelo.experimento(X, y, val_split=val_split, batch_size=batch_size, seed=seed, epochs=epochs)

    return history

def run_fif(fif_names, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
   
    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand",  "feet"])
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)
    print(f"Clases: {classes}\n")

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())
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
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand","rest"])
    X, y, classes = dataProvider.get_data()

    num_clases = len(classes)
    print(f"Clases: {classes}\n")

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = dataProvider.get_channel_names())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    history = modelo.finetuning(X_train, y_train, epochs=epochs, seed=seed, batch_size=batch_size)

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

def run_fif_piepline(fif_names, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    dataProvider = FifDataProvider(fif_paths = fif_names, annotations_names=["left_hand", "right_hand"])
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


if __name__ == "__main__":
    input_type = input("¿Cargar datos de MOABB[1], de archivos .fif[2] o de archivos .fif con pipeline separado[3] o archivos .fif con pipeline independiente[4]?: ").strip().lower()
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

        fif_names = ["EEG_controller_app/recordings/suj3_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_4_raw.fif"]

        run_fif(
            fif_names    = fif_names,
            epochs       = EPOCHS,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = 0.2,
            seed         = SEED,
        )

    elif input_type == "3":

        fif_names = ["EEG_controller_app/recordings/suj3_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_4_raw.fif"]

        run_fif_separado(
            fif_names    = fif_names,
            epochs       = 10,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = 0.6,
            seed         = SEED,
        )

    elif input_type == "4":

        fif_names = ["EEG_controller_app/recordings/suj3_1_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_2_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_3_raw.fif"]
        fif_names += ["EEG_controller_app/recordings/suj3_4_raw.fif"]

        run_fif_piepline(
            fif_names    = fif_names,
            epochs       = 10,
            batch_size   = BATCH_SIZE,
            lr           = LR,
            val_split    = 0.4,
            seed         = SEED,
        )