from sys import platform
from rich.table import Table
from rich.text import Text
from rich.progress import Progress
import time
import numpy as np

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

from utils import seleccionarPuertoCOM
from configuracion_canales import ChannelConfig
from visualizar_impedancias import visualizar_impedancias

def mapear_impedancias(lecturas: list[float], canales: list[ChannelConfig]) -> dict[str, float]:
    """
    Dado un listado de lecturas de impedancia (en el mismo orden que los canales habilitados)
    y la configuración de canales, devuelve un diccionario {nombre_electrodo: impedancia}.
    """
    resultado = {}
    idx = 0
    for ch in canales:
        if ch.enabled:
            resultado[ch.electrode] = lecturas[idx]
            idx += 1

    return resultado

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
    console.print("  2) Mostrar visualización de impedancias")
    console.print("  3) Volver al menú principal\n")

    opcion = console.input("Seleccione una opción: ").strip()

    while opcion not in ["1", "2", "3"]:
        console.print("[red]Opción no válida[/red]")
        opcion = console.input("Seleccione una opción: ").strip()

    return opcion

def medir_impedancias(puerto_com, canales, console):
    electrodes = {ch.index: ch.electrode for ch in canales if ch.enabled}
    imp = None

    eeg = acquisition.EEG()

    with EEGManager() as mgr:
        eeg.setup(
            mgr=mgr,
            port=puerto_com,
            cap=electrodes,
            gain=8,
            bias=[]
        )

        eeg.start_impedance_measurement()

        start_time = time.time()
        while time.time()-start_time < 20:
            time.sleep(1)
            imp = eeg.calc_impedances()
            console.print(imp)

        eeg.stop_impedance_measurement()
        mgr.disconnect()
    
    eeg.close()

    console.input("Medición de impedancias finalizada. Pulse Enter para continuar...")
    return imp


def medir_impedancias_interactivo(console, canales, mensajeSalida = "Volver al menú principal", puerto_com = None):
    console.clear()
    
    electrodosActivos = {ch.index: ch.electrode for ch in canales if ch.enabled}
    
    if not electrodosActivos:
        console.print("[red]No hay canales habilitados para la medición de impedancias.[/red]")
        console.input("Pulse Enter para continuar...")
        return None
    
    if puerto_com is None:
        puerto_com = seleccionarPuertoCOM(console)

    if puerto_com is None:
        return None
    
    opcion = ""
    imp = medir_impedancias(puerto_com=puerto_com, canales=canales, console=console)

    while opcion != "3":
        console.clear()
        tabla = build_channels_table(canales, imp)
        console.print(tabla)
        opcion = menu_post_impedancias(console)

        match opcion:
            case "1":
                imp = medir_impedancias(puerto_com=puerto_com, canales=canales, console=console)

            case "2":
                map_imp = mapear_impedancias(imp, canales)
                visualizar_impedancias(electrodos=map_imp,umbral_bueno=10, umbral_malo=100)

            case "3":
                console.print(mensajeSalida)