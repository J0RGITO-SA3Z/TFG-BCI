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


import serial.tools.list_ports

import numpy as np

class EEGRecorder:

    def __init__(self,mgr: EEGManager, console: Console):
        self.mgr = mgr
        self.console = console
        return
    

    async def start(
        self,
        channelsConfig: list
    ) -> mne.io.RawArray:
        """
        Graba EEG desde BrainAccess MIDI y devuelve un RawArray de MNE.
        La duración se pide por consola.
        """

        # -----------------------------
        # Pedir duración por consola
        # -----------------------------
        while True:
            try:
                record_sec = int(input("Introduce la duración de la grabación (en segundos): "))
                if record_sec <= 0:
                    raise ValueError
                break
            except ValueError:
                self.console.print("[red]Introduce un número entero válido (> 0)[/red]")

        self.console.print(f"🟢 Grabando EEG durante [bold]{record_sec}[/bold] segundos...")

        # -----------------------------
        # Configuración de canales
        # -----------------------------
        bias = [ch.index for ch in channelsConfig if ch.is_bias]
        electrodes = {ch.index: ch.electrode for ch in channelsConfig if ch.enabled}
        cap_15 = eeg_channel.EEGCap(electrodes=electrodes)

        # -----------------------------
        # Setup EEG
        # -----------------------------
        eeg = EEG(mode="accumulate")  # usa "roll" si luego quieres streaming
        eeg.setup(
            mgr=self.mgr,
            device_name=port,
            cap=cap_15,
            sfreq=sfreq,
            gain=8,
            bias=bias
        )

        # -----------------------------
        # Grabación
        # -----------------------------
        eeg.start_acquisition()
        time.sleep(record_sec)
        eeg.stop_acquisition()
        eeg.close()

        # -----------------------------
        # Exportar a MNE
        # -----------------------------
        raw = eeg.get_mne()

        out_fif = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
        raw.save(out_fif, overwrite=True)

        self.console.print(f"💾 EEG grabado y guardado en [green]{out_fif}[/green]")

        return raw
    
    