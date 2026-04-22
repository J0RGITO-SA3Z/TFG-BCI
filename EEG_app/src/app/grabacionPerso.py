import time
import mne

from app.EEGRecorder import EEGRecorder
from app.tcp.eeg_live_server import EEGLiveServer
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from app.app_utils import seleccionarPuertoCOM
from app.app_utils import RECORD_DIR

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

def startRecording(channelsConfig, acciones, console):
    """
    Graba EEG desde BrainAccess MIDI y devuelve un RawArray de MNE.
    La duración se pide por consola.
    """
    puerto_COM = seleccionarPuertoCOM(console)

    if puerto_COM is None:
        return

    eeg = EEGRecorder() 

    with EEGManager() as mgr:
        fileOutput = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
        fileOutput = console.input("Introduce el nombre del archivo de salida (sin extensión): ")
        fileOutput = fileOutput.strip() + ".fif"
        fileOutput = RECORD_DIR / fileOutput

        console.print(f"[bold]Grabando EEG hasta que se detenga la grabación...[/bold]")
        eeg.configAndConect(mgr=mgr, COM_port=puerto_COM, channelConfig=channelsConfig, gain=8)
        info = eeg.get_info()

        live_server = EEGLiveServer(
            ch_names=eeg.get_ch_names_ordered(),
            sfreq=eeg.get_sfreq(),
            ch_types=eeg.get_ch_types_ordered(),
        )
        live_server.start()
        eeg.register_callback(live_server.newChunk)

        eeg.iniciarGrabacion()

        live_server.sendInfo(
            ch_names=eeg.get_ch_names_ordered(),
            ch_types=eeg.get_ch_types_ordered(),
            sfreq=eeg.get_sfreq(),
        )

        entrada = -2
        
        while entrada != 1:
            #console.clear()
            console.print(build_recording_panel(acciones))
            
            entrada = console.input("[cyan]Seleccione una opción:[/] ")
            try:
                entrada = int(entrada)
            except ValueError:
                entrada = -2

            while entrada <0 or entrada >len(acciones)+1:
                entrada = console.input("[red]Índice inválido. Intente de nuevo:[/]")
                try:
                    entrada = int(entrada)
                except ValueError:
                    entrada = -2
                
            if entrada > 1 and entrada <= len(acciones)+1:
                eeg.anotar(acciones[entrada-2])

        eeg.detenerGrabacion()
        raw = eeg.get_mne()
        live_server.stop()
        eeg.desconectar()
        live_server.stop()

    raw.save(fileOutput, overwrite=True)

    console.print(f"\nEEG grabado y guardado en [green]{fileOutput}[/green]")
    eeg.data.mne_raw.plot(scalings='auto', verbose=False)
    console.input("[dim]Pulse Enter para continuar...[/dim]")
    
    eeg.cerrarLibreria()

    return raw