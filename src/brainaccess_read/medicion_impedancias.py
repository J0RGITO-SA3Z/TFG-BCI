from sys import platform
from rich.table import Table
from rich.text import Text
from rich.progress import Progress
import time
import numpy as np

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

from UI_utils import seleccionarPuertoCOM

def build_channels_table(canales, lecturas) -> Table:
    table = Table(title="Estado de Canales")

    table.add_column("Electrodo")
    table.add_column("Bias", justify="center")
    table.add_column("Impedancia (Ω)", justify="right")

    canales_enabled = [ch for ch in canales if ch.enabled]

    idx = 0
    
    for ch in canales_enabled:
        impedancia = f"{lecturas[idx]:.2f}"
        idx += 1

        table.add_row(
            ch.electrode,
            "[yellow]sí[/yellow]" if ch.is_bias else "No",
            impedancia
        )

    return table

def medir_impedancias(console, canales):
    console.clear()
    
    bias = [ch.index for ch in canales if ch.is_bias]
    electrodes = {ch.index: ch.electrode for ch in canales if ch.enabled}
    
    if not electrodes:
        console.print("[red]No hay canales habilitados para la medición de impedancias.[/red]")
        console.input("Pulse Enter para continuar...")
        return None
    
    puerto_com = seleccionarPuertoCOM(console)
    console.clear()
    
    if puerto_com is None:
        return None
    
    eeg = acquisition.EEG()
    
    with EEGManager() as mgr:
        
        eeg.setup(
            mgr=mgr,
            port=puerto_com,
            cap=electrodes,
            gain=12,
            bias=bias
        )
        
        eeg.start_impedance_measurement()
        
        with Progress() as progress:
            tarea = progress.add_task("Cargando...", total=100)

            for _ in range(100):
                time.sleep(0.05)  # 100 * 0.05 = 5 segundos
                progress.update(tarea, advance=1)
        
        
        imp = eeg.calc_impedances(4)
        eeg.stop_impedance_measurement()
    
    tabla = build_channels_table(canales, imp)
    console.print(tabla)
    pulse
        