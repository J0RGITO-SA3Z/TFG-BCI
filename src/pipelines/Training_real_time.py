import random
import numpy as np
from rich.console import Console
import time
import winsound

from brainaccess.utils.acquisition import EEG
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

from EEG_controller_app.src.utils import RECORD_DIR
from EEG_controller_app.src.EEGRecorder import EEGRecorder
from EEG_controller_app.src.eeg_live_server import EEGLiveServer
from EEG_controller_app.src.utils import seleccionarPuertoCOM
from EEG_controller_app.src.ventanaExperimentoVisual import ventanaExperimentoVisual


class Training_real_time:
    
    def __init__(self,console:Console):
        self.console = console
        self.tmpBaselineInicial = 30
        self.tmpBaselineEpoch = 2
        self.tmpBreack = 2
        self.tmpIM = 4
        self.numTrialsClase = 30
        return

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

    def start(self,channelsConfig,lista = ["IZQUIERDA","DERECHA","ABAJO","DESCANSO"]):
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

        # Calcular la matriz y hacer finetune
        raw.save(fileOutput, overwrite=True)
        self.console.clear()
        self.console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
        raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")
