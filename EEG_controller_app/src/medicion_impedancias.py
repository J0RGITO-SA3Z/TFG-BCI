from sys import platform
from rich.table import Table
from rich.text import Text
from rich.progress import Progress
import time
import numpy as np

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

from utils import seleccionarPuertoCOM

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

def menu_post_impedancias(console):
    console.print("Seleccione una opción:")
    console.print("  1) Repetir medición")
    console.print("  2) Volver al menú principal\n")

    opcion = ""

    while opcion not in ["1", "2"]:
        opcion = console.input("Seleccione una opción: ").strip()

        if opcion == "1":
            return opcion
        elif opcion == "2":
            return opcion
        else:
            console.print("[red]Opción no válida[/red]")

def medir_impedancias(console, canales):
    console.clear()
    
    bias = [ch.index for ch in canales if ch.is_bias]
    electrodes = {ch.index: ch.electrode for ch in canales if ch.enabled}
    
    if not electrodes:
        console.print("[red]No hay canales habilitados para la medición de impedancias.[/red]")
        console.input("Pulse Enter para continuar...")
        return None
    
    puerto_com = seleccionarPuertoCOM(console)
    
    if puerto_com is None:
        return None
    opcion = ""

    imp = None
    while opcion != "2":
        console.clear()
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
            # Print impedances
            start_time = time.time()
            while time.time()-start_time < 20:
                time.sleep(1)
                imp = eeg.calc_impedances()
                print(imp)

            # Stop measuring impedance
            eeg.stop_impedance_measurement()
            mgr.disconnect()
           
        tabla = build_channels_table(canales, imp)
        console.print(tabla)
        opcion = menu_post_impedancias(console)