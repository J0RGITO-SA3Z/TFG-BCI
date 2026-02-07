import os
from unittest import case
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

import json

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import asyncio

from UI_utils import ask_validated

from visualizacion_directo import EEGVisualizacionDirecto
from grabacion import EEGRecorder
from medicion_impedancias import medir_impedancias
from set_configuracion import channels_menu
from set_configuracion import ChannelConfig
    
DIR_ACCIONES = "acciones"

"""FUNCIONES PARA EL MENU DE ACCIONES"""

def validar_nombre_accion(console,text):
    if " " in text:
        console.print("[red]No se permiten espacios[/]")
        return False

    if len(text) > 20:
        console.print("[red]Máximo 20 caracteres[/]")
        return False

    return True

def validar_opciones(console, numero, text):
    try:
        opcion = int(text)
        if 1 <= opcion <= numero:
            return True
        else:
            console.print(f"[red]La opción debe estar entre 1 y {numero}[/]")
            return False
    except ValueError:
        console.print("[red]Debe introducir un número válido[/]")
        return False

def render_actions_table(actions):
    table = Table(title="Acciones registradas", expand=True)

    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Nombre de la acción", style="white")

    if not actions:
        table.add_row("-", "[italic dim]No hay acciones añadidas[/]")
    else:
        for idx, name in enumerate(actions):
            table.add_row(str(idx), name)

    return table

def anadir_accion(actions,console):
    name = ask_validated(console, "Nombre de la nueva acción [dim]-1 para cancelar[/dim]", lambda text: validar_nombre_accion(console, text))
    
    if name == "-1":
        return
    
    actions.append(name)
    
def eliminar_accion(actions,console):
    if not actions:
        return

    idx = -2
    while (idx < -1 or idx >= len(actions)):
        idx = IntPrompt.ask("ID de la acción a eliminar [dim](-1 para cancelar)[/dim]")
        
        if 0 <= idx < len(actions):
            actions.pop(idx)
        elif idx == -1:
            return
        else:
            console.print("[red]Índice fuera de rango[/]")
            
def ensure_actions_dir():
    if not os.path.exists(DIR_ACCIONES):
        os.makedirs(DIR_ACCIONES)
            
def guardar_acciones(acciones,console):
    ensure_actions_dir()
    
    archivo = Prompt.ask("Nombre del archivo (sin espacios) [dim](-1 para cancelar)[/dim]")
    
    if archivo == "-1":
        return
    
    archivo = f"{archivo}.json"
    archivo = os.path.join(DIR_ACCIONES, archivo)
    
    try:
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(acciones, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Acciones guardadas en {archivo}[/]")

    except Exception as e:
        console.print(f"[red]Error al guardar acciones: {e}[/]")
        
    console.input("[dim]Pulse Enter para continuar...[/dim]")
    
def listar_acciones_guardadas():
    ensure_actions_dir()
    
    return sorted(
        f for f in os.listdir(DIR_ACCIONES)
        if f.endswith(".json")
    )
    
def render_files_table(files):
    table = Table(title="Acciones disponibles", expand=True)
    table.add_column("ID", justify="center", style="cyan")
    table.add_column("Archivo")

    if not files:
        table.add_row("-", "[italic dim]No hay archivos[/]")
    else:
        for i, f in enumerate(files):
            table.add_row(str(i), f)

    return table

def cargar_acciones(console):
    archivos = listar_acciones_guardadas()
    console.print(render_files_table(archivos))

    if not archivos:
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return None

    idx = -2
    while (idx < -1 or idx >= len(archivos)):
        idx = IntPrompt.ask("ID del archivo a cargar [dim](-1 para cancelar)[/dim]")

        if idx == -1:
            return None

        if 0 <= idx < len(archivos):
            archivo = os.path.join(DIR_ACCIONES, archivos[idx])
            try:
                with open(archivo, "r", encoding="utf-8") as f:
                    acciones = json.load(f)
                console.print(f"[green]Acciones cargadas desde {archivo}[/]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
                return acciones
            except Exception as e:
                console.print(f"[red]Error al cargar acciones: {e}[/]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
                return []
        else:
            console.print("[red]Índice fuera de rango[/]")
            
    return None
            
def menu_acciones(console,acciones):
    choice = ""
    
    while choice != "5":
        console.clear()

        console.print(
            Panel(
                render_actions_table(acciones),
                title="MENÚ DE ACCIONES",
                border_style="cyan",
            )
        )

        console.print("[bold cyan]Opciones:[/]")
        console.print("  1) Añadir acción")
        console.print("  2) Eliminar acción")
        console.print("  3) Guardar acciones")
        console.print("  4) Cargar acciones")
        console.print("  5) Volver al menú principal")
        
        choice = ask_validated(console, "Seleccione una opción", lambda text: validar_opciones(console, 5, text))
        
        match choice:
            case "1":
                anadir_accion(acciones,console)
            case "2":
                eliminar_accion(acciones,console)
            case "3":
                guardar_acciones(acciones, console)
            case "4":
                acciones_aux = cargar_acciones(console)
                if acciones_aux is not None:
                    acciones.clear()
                    acciones.extend(acciones_aux)
            case "5":
                console.print("[green]Volviendo al menú principal...[/green]")
                break
    

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
    menu.append("  6) Salir\n")
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

def main_menu(console):

    json_file = "channels_headset_15.json"
    route_json = "src/Disposition/configs/" + json_file
    
    channels = [
        ChannelConfig(i, f"CH{i+1}", True, False, "Cz")
        for i in range(16)
    ]

    acciones = []    
    
    while True:
        console.clear()
        console.print(build_root_menu())

        choice = console.input(">> ")

        match choice:
            case "1":
                json_file = channels_menu(channels, console, json_file)
                console.print("[bold green]Configuración guardada correctamente[/bold green]")

            case "2":
                menu_acciones(console,acciones)
        
            case "3":
                medir_impedancias(console, channels)

            case "4":
                with EEGManager() as mgr:
                    recorder = EEGRecorder(mgr, console)
                    recorder.start(channels,acciones)

            case "5":
                with EEGManager() as mgr:
                    visualizador = EEGVisualizacionDirecto(mgr, console)
                    asyncio.run(visualizador.start(channels)) 

            case "6":
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