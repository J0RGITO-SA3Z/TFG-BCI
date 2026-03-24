import os
import sys
import torch
from sklearn.model_selection import train_test_split

import moabb

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

from utils.Performance_Viewer import PerformanceViewer
from model_interface.MiRepNetInterface import MiRepNetInterface
from DataProvider.FifDataProvider import FifDataProvider

from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.ClassEventRemover import ClassEventRemover
from epoch_processing.EpochEventRenamer import EpochEventRenamer

moabb.set_log_level("ERROR")
SEED = 42

def run_pipeline(dataProvider, model_interface, epochs, epoch_pipeline, validation_split=0.2, exclude_training_classes = None, rename_training_classes = None):
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

    viewer = PerformanceViewer()
    viewer.summary(final_val_acc)
    viewer.plot_fine_tune(final_val_acc)
    viewer.plot_downstream(preds_array, probs_array, Y_val)

def run_MiRepNet_pipeline(fif_paths, epochs = 10, validation_split=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProvider = FifDataProvider(fif_paths = fif_paths, annotations_names=["left_hand", "right_hand"])

    epoch_training_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, training_clases = ["left_hand", "right_hand"])

    run_pipeline(dataProvider, modelo, epochs, epoch_training_pipeline, validation_split)