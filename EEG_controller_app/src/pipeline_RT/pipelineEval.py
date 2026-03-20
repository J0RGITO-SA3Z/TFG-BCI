import os
import torch
from sklearn.model_selection import train_test_split

import moabb

from pipeline_utils.Performance_Viewer import PerformanceViewer
from model_interface.MiRepNetInterface import MiRepNetInterface
from DataProvider.FifDataProvider import FifDataProvider

from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.ClassEventRemover import ClassEventRemover

moabb.set_log_level("ERROR")

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
SEED = 42

def run_pipeline(dataProvider, model_interface, epochs, epoch_pipeline, validation_split=0.2, exclude_training_classes = None):
    torch.manual_seed(SEED)
    X, Y, classes = dataProvider.get_data()

    # Dividir en train y validacion
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=validation_split, random_state=SEED, stratify=Y
    )

    # Excluir clases no usadas para el entrenamiento (ej: "rest")
    if exclude_training_classes is not None:
        remover = ClassEventRemover(exclude_training_classes)
        X_train, Y_train = remover.process_np(X_train, Y_train, shuffle=False)

    # Normalizacion de los datos
    X_train, Y_train = epoch_pipeline.process_np(X_train, Y_train,shuffle=False)
    X_val, Y_val = epoch_pipeline.process_np(X_val, Y_val,shuffle=False)

    # Finetuning del modelo
    final_val_acc = model_interface.finetuning(X_train, Y_train, X_val, Y_val, epochs=epochs)

    # Prediccion de las probabilidades de cada clase para cada muestra del set de validacion
    preds_array, probs_array = model_interface.predict_batch(X_val)

    viewer = PerformanceViewer()
    viewer.summary(final_val_acc)
    viewer.plot_fine_tune(final_val_acc)
    viewer.plot_downstream(preds_array, probs_array, Y_val)

def run_MiRepNet_pipeline(fif_paths, epochs = 10, validation_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataProvider = FifDataProvider(fif_paths = fif_paths, annotations_names=["left_hand", "right_hand", "feet", "rest"])

    epoch_training_pipeline = EpochProcessorPipeline([
        EuclideanAlignment(),         # alineamiento euclídeo (EA)
        SpatialInterpolator(actual_channel_positions = dataProvider.get_channel_names()),          # interpola/reordena canales a la topología objetivo 
    ])

    modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, training_clases = ["left_hand", "right_hand", "feet", "rest"])

    run_pipeline(dataProvider, modelo, epochs, epoch_training_pipeline, validation_split, exclude_training_classes="rest")