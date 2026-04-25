import os
import sys
import torch
from sklearn.model_selection import train_test_split

import moabb

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)

from util.Performance_Viewer import PerformanceViewer
from components.ModelInterface.MiRepNetInterface import MiRepNetInterface
from EEG_app.src.components.DataProvider.FifDataProvider import FifDataProvider

from components.EpochProcessing.EpochProcessorPipeline import EpochProcessorPipeline
from components.EpochProcessing.SpatialInterpolator import SpatialInterpolator
from components.EpochProcessing.EuclideanAlignment import EuclideanAlignment
from components.EpochProcessing.ClassEventRemover import ClassEventRemover
from components.EpochProcessing.EpochEventRenamer import EpochEventRenamer
from components.EpochProcessing.BadChannelInterpolator import BadChannelInterpolator

from components.EpochProcessing.BadChannelDetectors.AmplitudeThresholdDetector import AmplitudeThresholdDetector
from components.EpochProcessing.BadChannelDetectors.VarianceDetector import VarianceDetector
from components.EpochProcessing.BadChannelDetectors.GradientDetector import GradientDetector

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

moabb.set_log_level("ERROR")
SEED = 42

def run_pipeline(dataProvider, model_interface, epochs, epoch_pipeline, validation_split=0.2, exclude_training_classes = None, rename_training_classes = None, show_plots=True):
    torch.manual_seed(SEED)
    X, Y, _ = dataProvider.get_data()

    # Dividir en train y validacion
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=validation_split, random_state=SEED, stratify=Y
    )

    X_val2 , Y_val2 = None, None

    # Excluir clases no usadas para el entrenamiento (ej: "rest")
    if exclude_training_classes is not None:
        remover = ClassEventRemover(exclude_training_classes)
        X_train, Y_train = remover.process_np(X_train, Y_train)
        X_val2, Y_val2 = remover.process_np(X_val, Y_val)
    else :
        X_val2, Y_val2 = X_val, Y_val

    if rename_training_classes is not None:
        renamer = EpochEventRenamer(rename_training_classes)
        X_train, Y_train = renamer.process_np(X_train, Y_train)
        X_val2, Y_val2 = renamer.process_np(X_val2, Y_val2)

    # Normalizacion de los datos
    X_train, Y_train = epoch_pipeline.process_np(X_train, Y_train,shuffle=False)
    X_val, Y_val = epoch_pipeline.process_np(X_val, Y_val,shuffle=False)
    X_val2, Y_val2 = epoch_pipeline.process_np(X_val2, Y_val2,shuffle=False)

    # Finetuning del modelo
    final_val_acc = model_interface.finetuning(X_train, Y_train, X_val2, Y_val2, epochs=epochs)

    # Prediccion de las probabilidades de cada clase para cada muestra del set de validacion
    preds_array, probs_array = model_interface.predict_batch(X_val)

    preds_array, probs_array = model_interface.predict_batch(X_val)

    # 1. Calculamos las métricas clave
    acc = accuracy_score(Y_val, preds_array)
    prec = precision_score(Y_val, preds_array, average='macro', zero_division=0)
    rec = recall_score(Y_val, preds_array, average='macro', zero_division=0)
    f1 = f1_score(Y_val, preds_array, average='macro', zero_division=0)

    # 2. Mostramos las gráficas SOLO si show_plots es True
    if show_plots:
        viewer = PerformanceViewer()
        viewer.summary(final_val_acc)
        viewer.plot_fine_tune(final_val_acc)
        viewer.plot_downstream(preds_array, probs_array, Y_val)

    # 3. Devolvemos un diccionario con los resultados
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

def run_MiRepNet_pipeline(fif_paths, epochs = 10, validation_split=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProvider = FifDataProvider(fif_paths = fif_paths, annotations_names=["left_hand", "right_hand"])

    epoch_training_pipeline = EpochProcessorPipeline([
        BadChannelInterpolator(channels_max=4, print_history=True, actual_channel_positions=dataProvider.get_channel_names(), detectors= [
            AmplitudeThresholdDetector(threshold=100),                  # umbral de amplitud (ej: 100 microvoltios)
            VarianceDetector(threshold=1000.0, dead_threshold=1e-10),   # µV²
            GradientDetector(threshold=25.0)                            # µV/muestra a 250 Hz
        ]),  # interpolación de canales malos
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, training_clases = ["left_hand", "right_hand"])

    run_pipeline(dataProvider, modelo, epochs, epoch_training_pipeline, validation_split)
    
def main():
    fif_names = ["EEG_app/recordings/experimento_visual/suj1/suj1_1_raw.fif"]
    fif_names += ["EEG_app/recordings/experimento_visual/suj1/suj1_2_raw.fif"]
    fif_names += ["EEG_app/recordings/experimento_visual/suj1/suj1_3_raw.fif"]
    fif_names += ["EEG_app/recordings/experimento_visual/suj1/suj1_4_raw.fif"]

    run_MiRepNet_pipeline(fif_paths    = fif_names)

if __name__ == "__main__":
    main()