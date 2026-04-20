from sys import platform
from rich.table import Table
from rich.text import Text
from rich.progress import Progress

from app.app_utils import seleccionarPuertoCOM
from app.configuracion_canales import ChannelConfig
from app.visualizar_impedancias import visualizar_impedancias
from components.EEG_BrainAccess.ImpedanceMeter import ImpedanceMeter

def mapear_impedancias(lecturas: list[float], canales: list[ChannelConfig]) -> dict[str, float]:
    return ImpedanceMeter(puerto_com="", canales=canales).to_map(raw=lecturas)

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
    meter = ImpedanceMeter(puerto_com=puerto_com, canales=canales)
    imp = meter.measure(console=console)
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