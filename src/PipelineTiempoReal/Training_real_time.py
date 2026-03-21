import random
import numpy as np
from rich.console import Console
import time
import winsound
import torch
import os
import sys
from sklearn.model_selection import train_test_split

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── BrainAccess ─────────────────────────────────────────────────────────────────────
from brainaccess.utils.acquisition import EEG
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

# ── Imports modelInterface ──────────────────────────────────────────────────────────
from model_interface.MiRepNetInterface import MiRepNetInterface

# ── Imports de scripts de la aplicación ──────────────────────────────────────────────────────────
from EEG_controller_app.src.app_utils import RECORD_DIR
from EEG_controller_app.src.EEGRecorder import EEGRecorder
from EEG_controller_app.src.eeg_live_server import EEGLiveServer
from EEG_controller_app.src.app_utils import seleccionarPuertoCOM
from EEG_controller_app.src.ventanaExperimentoVisual import ventanaExperimentoVisual

# ── Imports visualización ─────────────────────────────────────────────────────
from utils.Performance_Viewer import PerformanceViewer

# ── Data Processing ─────────────────────────────────────────────────────────────────────
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from epoch_processing.SpatialInterpolator import SpatialInterpolator
from epoch_processing.EuclideanAlignment import EuclideanAlignment
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline

from src.DataProvider.FifDataProvider import LABEL_MAP
from epoch_processing.EpochProcessorPipeline import EpochProcessorPipeline
from raw_processing.RawProcessorPipeline import RawProcessorPipeline
from raw_processing.BandpassFilter import BandpassFilter
from raw_processing.AnnotationRenamer import AnnotationRenamer
from raw_processing.NotchFilter import NotchFilter
from raw_processing.Resampler import Resampler

from DataProvider.FifDataProvider import _raw_to_epochs 
from src.DataProvider.FifDataProvider import LABEL_MAP


class Training_real_time:
    
    def __init__(self,console:Console):
        self.console = console
        self.tmpBaselineInicial = 30
        self.tmpBaselineEpoch = 2
        self.tmpBreack = 2
        self.tmpIM = 4
        self.numTrialsClase = 30
        self.channelsnames = None
        self.matrix = None
        self.modelo = None
        return

    # Métodos privados de la clase ─────────────────────────────────────────────────────────────────────

    def __generar_lista(self,acciones, total): #Podría ser un método estático, no depende de la instancia
            n = len(acciones)
            if total % n != 0:
                raise ValueError("El total debe ser múltiplo del número de acciones")

            lista = acciones * (total // n)
            random.shuffle(lista)
            return lista
    
    def __trialToText(self, trial): # Podría ser un método estático, no depende de la instancia
        if trial == "IZQUIERDA":
            return "←"
        elif trial == "DERECHA":
            return "→"
        elif trial == "ABAJO":
            return "↓"
        elif trial == "DESCANSO":
            return "NADA"
        else:
            return trial
        
    # Cálculo de la matriz media de las matrizes de covarianza de cada trial seccion sacada de utils de MIRepNet
    def EA_Matrix(self, x):
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

    def preprocess(self, raw,annotations_names):
        '''
        Parameters
        ----------
        raw : mne.io.Raw
            Raw obtenido de la grabación con BrainAccess
        annotations_names : list of str
            Lista de nombres de anotaciones a renombrar en el pipeline de raw, por ejemplo ["left_hand", "right_hand", "feet"]

        Returns
        ----------
        epochs : mne.Epochs
            Epochs obtenidos tras procesar el raw y convertirlo a epochs, listos para usar en el fine-tuning del modelo.    
        '''

        # Extraemos los nombres de olos canales y los guardamos en un atributo de lista
        raw_types = raw.pick_types(eeg=True)
        self.channelsnames  = [ a.upper() for a in raw_types.ch_names]

        _raw_pipeline = RawProcessorPipeline([
                    # NotchFilter(50.0),
                    BandpassFilter(8.0, 30.0),
                    AnnotationRenamer(LABEL_MAP),
                    #CARReference(),
                    # Resampler(250),
                    # ICAProcessor(),
        ])

        raw = self._raw_pipeline.process(raw)
        epochs = _raw_to_epochs(raw, anotationsNames = self._annotations_names)

        return epochs

    # Función de fine-tune similar a la del script Pipeline para fif y MOABB
    def run_finetuning_pipeline(self, data_epochs , epochs, seed, annotations_names = ["left_hand", "right_hand", "feet"]):
        '''
        Parameters
        ----------
        data_epochs : mne.Epochs
            Epochs obtenidos tras procesar el raw y convertirlo a epochs, listos para usar en el fine-tuning del modelo.
        epochs : int
            Número de épocas para el fine-tuning del modelo.
        seed : int
            Semilla para la reproducibilidad del split entre train y test.
        annotations_names : list of str
            Lista de nombres de anotaciones a usar para el fine-tuning, por ejemplo ["left_hand", "right_hand", "feet"].

        Returns
        ----------
        None'''
        
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}\n")

        X = data_epochs.get_data()
        true_labels_numeric = data_epochs.events[:, 2]
        inv_event_id = {v: k for k, v in data_epochs.event_id.items()}
        true_labels = [inv_event_id[i] for i in true_labels_numeric]

        classes = sorted(set(true_labels))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[l] for l in true_labels], dtype=np.int64)


        epoch_pipeline = EpochProcessorPipeline([
            EuclideanAlignment(),         # alineamiento euclídeo (EA)
            SpatialInterpolator(actual_channel_positions = self.channelsnames),        # interpola/reordena canales a la topología objetivo 
        ])

        num_clases = len(classes)

        self.modelo = MiRepNetInterface(device=device, weight_path=WEIGHT_PATH, num_clases = num_clases, channels_names = self.channelsnames)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=y
        )

        X_train, y_train = epoch_pipeline.process_np(X_train, y_train,shuffle=False)
        X_val, y_val = epoch_pipeline.process_np(X_val, y_val,shuffle=False)

        historico = self.modelo.finetuning_processed(X_train, y_train, epochs=epochs)
        preds_array, probs_array = self.modelo.predict_batch_preprocessed(X_val)

        viewer = PerformanceViewer()
        viewer.summary(historico)
        viewer.plot_downstream2(probs_array, y_val, class_names = classes)


    def experimento_visual(self,channelsConfig,lista = ["left_hand", "right_hand"]):
        """
        Graba EEG desde BrainAccess MIDI y devuelve un RawArray de MNE.
        La duración se pide por consola.
        """
        self.console.clear()
        self.console.print("En el modo experimento visual, las acciones anotadas siempre son hizquierda, derecha, abajo, descanso. Si quieres anotaciones personalizadas, usa el modo de grabación manual.")
        self.console.input("Pulse Enter para continuar...")
        self.console.clear()

        puerto_COM = seleccionarPuertoCOM(self.console)

        if puerto_COM is None:
            return

        raw = None
        eeg = EEGRecorder()
        with EEGManager() as mgr:
            eeg.configAndConect(mgr=mgr, COM_port=puerto_COM, channelConfig=channelsConfig, gain=8)

            self.console.print("[green]EEG configurado correctamente[/green]\n")

            self.numnumTrialsClase= self.numTrialsClase * lista.__len__()
            trials = self.__generar_lista(lista, self.numTrials)
            print(trials)
            
            live_server = EEGLiveServer(
                ch_names=eeg.get_ch_names_ordered(),
                sfreq=eeg.get_sfreq(),
                ch_types=eeg.get_ch_types_ordered(),
                total_epochs= self.numTrials,
                initial_action="Empezando"
            )
            live_server.start()
            eeg.register_callback(live_server.newChunk)

            ventana = ventanaExperimentoVisual()
            ventana.open()

            eeg.iniciarGrabacion()

            ventana.draw_text("Baseline")
            live_server.setAction("Baseline")
            time.sleep(self.tmpBaselineInicial)
            ventana.draw_text("Concéntrate")
            live_server.setAction("Concéntrate")

            time.sleep(9.5)
            winsound.Beep(1000, 500)

            for trial in trials:
                live_server.setAction(trial)
                live_server.increaseEpoch()
                ventana.draw_text("+")
                eeg.anotar("CROSS")
                time.sleep(self.tmpBaselineEpoch-0.5)
                winsound.Beep(1000, 500)

                ventana.draw_text(self.__trialToText(trial))
                eeg.anotar(trial)

                time.sleep(self.tmpIM)
                ventana.draw_text("")
                eeg.anotar("BLANK")

                time.sleep(self.tmpBreack)
                
            raw = eeg.get_mne()
            eeg.detenerGrabacion()
            mgr.disconnect()
            live_server.stop()
            ventana.close()

        eeg.cerrarLibreria()

        self.console.print(f"\n[green]Grabación de entrenamiento finalizada[/green]")

        return raw

    # Métodos Publicos de la clase  ─────────────────────────────────────────────────────────────────────

    def getMatrix(self):
        return self.matrix
    
    def getModelo(self):
        return self.modelo

    def start(self, channelsConfig,lista = ["left_hand", "right_hand"]):

        # Comenzamos la grabación y sacamos raw 
        raw = self.experimento_visual(channelsConfig, lista)

        # Preprocesamos raw y convertimos a epochs de mne
        data_epochs = self.preprocess(raw,lista)

        # Extraemos las mediciones de la grabación y sacamos la matriz media de covarianzas
        data = data_epochs.get_data()
        self.matrix = self.EA_Matrix(data)

        # Hacemos fine-tune del modelo con los epochs de la grabación
        self.run_finetuning_pipeline(data_epochs, epochs=10, annotations_names = lista)

        return self.matrix, self.modelo
    