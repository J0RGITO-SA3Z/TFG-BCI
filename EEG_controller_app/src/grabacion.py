import time
import mne

from brainaccess.utils.acquisition import EEG
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from utils import seleccionarPuertoCOM
from utils import lee_indice

def build_recording_panel(acciones):
    header = Columns()
    title = "MENÚ DE GRABACIÓN"
    status = Text("EEG GRABANDO", style="bold green")
    status.justify = "right"

    header = Columns(
        [title, status],
        expand=True
    )
    
    subtitle = Text("Permite el control de la grabación y la inserción de eventos.", style="dim")

    body = Text()
    body.append("1) Acabar grabación\n\n", style="bold")
    body.append("Eventos:\n", style="bold cyan")
    
    for i, accion in enumerate(acciones, start=2):
        body.append(f"  {i}) {accion}\n")

    content = Group(
        header,
        subtitle,
        Text(""),
        body
    )

    return Panel(
        content,
        border_style="green",
        padding=(1, 2),
    )

class EEGRecorder:

    def __init__(self,mgr: EEGManager, console: Console):
        self.mgr = mgr
        self.console = console
        return
    

    def start(self,channelsConfig,acciones):
        """
        Graba EEG desde BrainAccess MIDI y devuelve un RawArray de MNE.
        La duración se pide por consola.
        """

        bias = [ch.index for ch in channelsConfig if ch.is_bias]
        electrodes = {ch.index: ch.electrode for ch in channelsConfig if ch.enabled}
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
        
        fileOutput = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
        fileOutput = self.console.input("Introduce el nombre del archivo de salida (sin extensión): ")

        self.console.print(f"[bold]Grabando EEG hasta que se detenga la grabación...[/bold]")

        eeg.start_acquisition()
        entrada = -2
        
        while entrada != 1:
            self.console.clear()
            self.console.print(build_recording_panel(acciones))
            
            entrada = self.console.input("[cyan]Seleccione una opción:[/] ")
            try:
                entrada = int(entrada)
            except ValueError:
                entrada = -2

            while entrada <0 or entrada >len(acciones)+1:
                entrada = self.console.input("[red]Índice inválido. Intente de nuevo:[/]")
                try:
                    entrada = int(entrada)
                except ValueError:
                    entrada = -2
                
                
            if entrada > 1 and entrada <= len(acciones)+1:
                eeg.annotate(acciones[entrada-2])
            
        eeg.stop_acquisition()
        eeg.close()

        raw = eeg.get_mne()
        raw.save(fileOutput, overwrite=True)

        self.console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
        eeg.data.mne_raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")

        return raw