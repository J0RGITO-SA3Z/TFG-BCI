import time
import mne
import random

from brainaccess.utils.acquisition import EEG
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from rich.prompt import IntPrompt
from UI_utils import seleccionarPuertoCOM
from UI_utils import lee_indice
from ventanaExperimentoVisual import ventanaExperimentoVisual


class ExperimentoVisual:

    def __init__(self,mgr: EEGManager, console: Console):
        self.mgr = mgr
        self.console = console
        self.tmpBaselineInicial = 30
        self.tmpBaselineEpoch = 2
        self.tmpBreack = 2
        self.tmpIM = 4
        self.numTrials = 60
        return
    
    def __generar_lista(self,acciones, total):
        n = len(acciones)
        if total % n != 0:
            raise ValueError("El total debe ser múltiplo del número de acciones")

        lista = acciones * (total // n)
        random.shuffle(lista)
        return lista
    
    def __trialToText(self, trial):
        if trial == "IZQUIERDA":
            return "<<<"
        elif trial == "DERECHA":
            return ">>>"
        elif trial == "ABAJO":
            return "VVV"
        elif trial == "DESCANSO":
            return "NADA"
        else:
            return trial
    

    def start(self,channelsConfig):
        """
        Graba EEG desde BrainAccess MIDI y devuelve un RawArray de MNE.
        La duración se pide por consola.
        """

        bias = [ch.index for ch in channelsConfig if ch.is_bias]
        electrodes = {ch.index: ch.electrode for ch in channelsConfig if ch.enabled}

        self.console.clear()
        self.console.print("En el modo experimento visual, las acciones anotadas siempre son hizquierda, derecha, abajo, descanso. Si quieres anotaciones personalizadas, usa el modo de grabación manual.")
        self.console.input("Pulse Enter para continuar...")
        self.console.clear()

        puerto_COM = seleccionarPuertoCOM(self.console)

        eeg = EEG(mode="accumulate")
        eeg.setup(
            mgr=self.mgr,
            port=puerto_COM,
            cap=electrodes,
            gain=8,
            bias=bias
        )

        self.console.print("[green]EEG configurado correctamente[/green]\n")

        self.numTrials = IntPrompt.ask("Introduce el numero de trials por clase (4 clases y 8s por trial)")
        self.numTrials = self.numTrials * 4
        trials = self.__generar_lista(["IZQUIERDA","DERECHA","ABAJO","DESCANSO"], self.numTrials)

        fileOutput = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
        fileOutput = self.console.input("Introduce el nombre del archivo de salida (sin extensión): ")
        fileOutput += ".fif"

        self.console.print(f"[bold]Grabando EEG hasta que se detenga la grabación...[/bold]")

        experimentoVisual = ventanaExperimentoVisual()
        experimentoVisual.open()

        eeg.start_acquisition()

        self.console.print("Baseline")
        experimentoVisual.draw_text("Baseline")
        time.sleep(self.tmpBaselineInicial)
        self.console.print("Concéntrate")
        experimentoVisual.draw_text("Concéntrate")
        time.sleep(10)
        self.console.print("YA")

        for trial in trials:
            experimentoVisual.draw_text("+")
            eeg.annotate("CROSS")
            time.sleep(self.tmpBaselineEpoch)

            experimentoVisual.draw_text(self.__trialToText(trial))
            eeg.annotate(trial)

            time.sleep(self.tmpIM)
            experimentoVisual.draw_text("")
            eeg.annotate("BLANK")

            time.sleep(self.tmpBreack)
            
        eeg.stop_acquisition()
        eeg.close()
        experimentoVisual.close()

        raw = eeg.get_mne()
        raw.save(fileOutput, overwrite=True)

        self.console.clear()
        self.console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
        eeg.data.mne_raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")

        return raw