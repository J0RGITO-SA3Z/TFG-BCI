from rich.console import Console
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import asyncio

from visualizacion_directo import EEGVisualizacionDirecto
from ExperimentoVisual import ExperimentoVisual
from grabacion import EEGRecorder
from medicion_impedancias import medir_impedancias_interactivo
from configuracion_canales import channels_menu
from configuracion_canales import load_default_channel_config
from configuracion_canales import ChannelConfig
from configuracion_acciones import menu_acciones
from configuracion_acciones import load_default_actions
    
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
    menu.append("  3) Test impedancias\n")
    menu.append("  4) Grabar\n")
    menu.append("  5) Tiempo real\n")
    menu.append("  6) Experimento Visual\n")
    menu.append("  7) Salir\n")
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
                medir_impedancias_interactivo(console, channels)

            case "4":
                with EEGManager() as mgr:
                    recorder = EEGRecorder(mgr, console)
                    recorder.start(channels,acciones)

            case "5":
                with EEGManager() as mgr:
                    visualizador = EEGVisualizacionDirecto(mgr, console)
                    asyncio.run(visualizador.start(channels)) 
            
            case "6":
                with EEGManager() as mgr:
                    experimento = ExperimentoVisual(mgr, console)
                    experimento.start(channels)

            case "7":
                console.print("[bold green]Saliendo del sistema BCI[/bold green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

def main():
    console = Console()

    bacore.init(bacore.Version(2, 0, 0))

    main_menu(console)

if __name__ == "__main__":
    main()