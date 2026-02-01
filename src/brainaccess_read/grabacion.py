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
from UI_utils import seleccionarPuertoCOM


import serial.tools.list_ports

import numpy as np

class EEGRecorder:

    def __init__(self,mgr: EEGManager, console: Console):
        self.mgr = mgr
        self.console = console
        return
    

    def start(self,channelsConfig):
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
            gain=12,
            bias=bias
        )

        self.console.print("[green]EEG configurado correctamente[/green]\n")

        while True:
            try:
                record_sec = int(self.console.input("Introduce la duración de la grabación (en segundos): "))
                if record_sec <= 0:
                    raise ValueError
                break
            except ValueError:
                self.console.print("[red]Introduce un número entero válido (> 0)[/red]")
        
        fileOutput = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
        fileOutput = self.console.input("Introduce el nombre del archivo de salida (sin extensión): ")

        self.console.print(f"Grabando EEG durante [bold]{record_sec}[/bold] segundos...")

        eeg.start_acquisition()
        time.sleep(record_sec)
        eeg.stop_acquisition()
        eeg.close()

        raw = eeg.get_mne()
        raw.save(fileOutput, overwrite=True)
        

        self.console.print(f"EEG grabado y guardado en [green]{fileOutput}[/green]")
        eeg.data.mne_raw.filter(1, 40).plot(scalings='auto', verbose=False)
        self.console.input("[dim]Pulse Enter para continuar...[/dim]")

        return raw