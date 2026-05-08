from rich.console import Console
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns

import asyncio
import os, sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from app.visualizacion_directo import visualizacion_directo
from app.ExperimentoVisual import ExperimentoVisual
from app.grabacion import startRecording
from app.medicion_impedancias import medir_impedancias_interactivo
from app.ver_bateria import ver_bateria
from app.configuracion_canales import channels_menu
from app.configuracion_canales import load_default_channel_config
from app.configuracion_canales import ChannelConfig
from app.configuracion_acciones import menu_acciones
from app.configuracion_acciones import load_default_actions
from app.validacion_sujeto import evaluar_sujeto_MiRepNet
from app.experimento_RT import interfaz_experimento_RT
from app.visualizar_fif import visualizar_fif
    
DIR_ACCIONES = "acciones"

"""FUNCIONES DEL MENU PRINCIPAL DE LA APLICACION"""

def build_root_menu():
    # ---- Línea superior: título + estado ----
    title = Text("Menú Principal del Sistema BCI", style="bold")
    subtitle = Text("Entorno de adquisición, configuración y visualización EEG. Introduce el indice de la opcion a ejecutar", style="dim")

    header = Columns(
        [title],
        expand=True
    )

    # ---- Opciones ----
    menu = Text()

    menu.append("  1) Ver / editar configuración de canales\n")
    menu.append("  2) ver / editar lista de acciones\n")
    menu.append("  3) Ver batería del EEG\n")
    menu.append("  4) Test impedancias\n")
    menu.append("  5) Grabar\n")
    menu.append("  6) Tiempo real\n")
    menu.append("  7) Experimento Visual\n")
    menu.append("  8) Evaluar sujeto MiRepNet\n")
    menu.append("  9) Experimento Tiempo real\n")
    menu.append("  10) Visualizar FIFs\n")
    menu.append("  11) Salir\n")
    content = Group(
        header,
        subtitle,
        Text(""),
        menu
    )

    return Panel(
        content,
        border_style="white",
        padding=(1, 2)
    )

def inicializar_aplicacion(acciones, channels):
    salida_channel = load_default_channel_config(channels)
    salida_acciones = load_default_actions(acciones)

    if salida_channel == -1:
        channels_aux = [
            ChannelConfig(i, f"CH{i+1}", True, False, "Cz")
            for i in range(16)
        ]

        channels.clear()
        channels.extend(channels_aux)

    if salida_acciones == -1:
        acciones.clear()

def main_menu(console):
    acciones = []    
    channels = []

    inicializar_aplicacion(acciones, channels)
    
    while True:
        console.clear()
        console.print(build_root_menu())

        choice = console.input(">> ")

        match choice:
            case "1":
                channels_menu(channels, console)
                console.print("[bold green]Configuración guardada correctamente[/bold green]")

            case "2":
                menu_acciones(console,acciones)

            case "3":
                ver_bateria(console)

            case "4":
                medir_impedancias_interactivo(console, channels)

            case "5":
                startRecording(channels, acciones, console)

            case "6":
                asyncio.run(visualizacion_directo(channels, console))

            case "7":
                experimento = ExperimentoVisual(console)
                experimento.start(channels)

            case "8":
                evaluar_sujeto_MiRepNet(console)

            case "9":
                interfaz_experimento_RT(channels, console)

            case "10":
                visualizar_fif(console)

            case "11":
                console.print("[bold green]Saliendo del sistema BCI[/bold green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

def main():
    console = Console()

    main_menu(console)

if __name__ == "__main__":
    main()