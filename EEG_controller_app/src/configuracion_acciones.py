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
from pathlib import Path

from utils import ask_validated
from utils import ACTIONS_CONF_DIR, DEFAULT_ACTIONS_CONFIG_FILE

## Valida que el nombre de la accion no contenga espacios y tenga un máximo de 20 caracteres
def validar_nombre_accion(console,text):
    if " " in text:
        console.print("[red]Valor invalido. No se permiten espacios[/]")
        return False

    if len(text) > 20:
        console.print("[red]Valor invalido. Máximo 20 caracteres[/]")
        return False

    return True

def ensure_actions_dir():
    Path(ACTIONS_CONF_DIR).mkdir(parents=True, exist_ok=True)
    
## Funcion para añadir una accion a la lista de acciones.
def anadir_accion(actions,console):
    name = ask_validated("Nombre de la nueva acción [dim]-1 para cancelar[/dim]", lambda text: validar_nombre_accion(console, text))
    
    if name == "-1":
        return
    
    actions.append(name)

## Función para eliminar una acción de la lista de acciones.
def eliminar_accion(actions,console):
    if not actions:
        return

    idx = -2
    while (idx < -1 or idx >= len(actions)):
        idx = IntPrompt.ask("ID de la acción a eliminar [dim](-1 para cancelar)[/dim]")
        
    if idx == -1:
        return 
    
    actions.pop(idx)

## Función encargada de guarda las acciones en un archivo json en el archivo especificado
def guardar_acciones(acciones, archivo):
    ensure_actions_dir()
    
    try:
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(acciones, f, indent=2, ensure_ascii=False)

        return 0

    except Exception as e:
       return 1

## Interfaz interactiva para guardas las acciones en un archivo json dentro de la carpeta acciones.
def guardar_acciones_interactivo(acciones,console):
    console.clear()
    archivo = Prompt.ask("Nombre del archivo (sin extension) [dim](-1 para cancelar)[/dim]")
    
    if archivo == "-1":
        return
    
    archivo = f"{archivo}.json"
    archivo = os.path.join(ACTIONS_CONF_DIR, archivo)

    salida = guardar_acciones(acciones, archivo)

    if salida == 0:
        console.print(f"[green]Acciones guardadas en {archivo}[/]")
    else:
        console.print(f"[red]Error al guardar acciones[/]")
        
    console.input("[dim]Pulse Enter para continuar...[/dim]")

## Crea una lista con los archivo json que hay en la carpeta acciones.
def listar_acciones_guardadas():
    ensure_actions_dir()
    
    return sorted(
        f for f in os.listdir(ACTIONS_CONF_DIR)
        if f.endswith(".json")
    )

## Genera una tabla con los archivos de acciones que se le pasen.
def build_files_table(files):
    table = Table(title="Acciones disponibles", expand=True)
    table.add_column("ID", justify="center", style="cyan")
    table.add_column("Archivo")

    if not files:
        table.add_row("-", "[italic dim]No hay archivos[/]")
    else:
        for i, f in enumerate(files):
            table.add_row(str(i), f)

    return table

## Interfaz interactica para cargar acciones desde un archivo json.
def cargar_acciones(console):
    console.clear()
    archivos = listar_acciones_guardadas()
    console.print(build_files_table(archivos))

    if not archivos:
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return None

    idx = -2
    while (idx < -1 or idx >= len(archivos)):
        idx = IntPrompt.ask("ID del archivo a cargar [dim](-1 para cancelar)[/dim]")

        if idx == -1:
            return None

        if 0 <= idx < len(archivos):
            archivo = os.path.join(ACTIONS_CONF_DIR, archivos[idx])
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

def load_default_actions(acciones):
    if not os.path.exists(DEFAULT_ACTIONS_CONFIG_FILE):
        return -1

    try:
        with open(DEFAULT_ACTIONS_CONFIG_FILE, "r", encoding="utf-8") as f:
            acciones_default = json.load(f)
            acciones.clear()
            acciones.extend(acciones_default)
            return 0
    except Exception as e:
        return -1
    
def set_default_actions(console,acciones):
    console.clear()
    guardar_acciones(acciones, DEFAULT_ACTIONS_CONFIG_FILE)
    console.print(f"[green]Las acciones actuales se han establecido como las acciones por defecto.[/green]")
    console.input("[green]Esto implica que al iniciar el programa se cargarán automáticamente.[/]")
    console.input("[dim]Pulse Enter para continuar...[/dim]")

##########################################################
##                                                      ##
##         MENU DE CONFIGURACIÓN DE ACCIONES            ##
##                                                      ##
##########################################################

## Genera la tabla de las acciones actuales
def build_actions_table(actions):
    table = Table(title="Acciones registradas", expand=True)

    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Nombre de la acción", style="white")

    if not actions:
        table.add_row("-", "[italic dim]No hay acciones añadidas[/]")
    else:
        for idx, name in enumerate(actions):
            table.add_row(str(idx), name)

    return table

## Función principal del menú de acciones
def menu_acciones(console,acciones):
    choice = ""
    
    while choice != "5":
        console.clear()

        console.print(
            Panel(
                build_actions_table(acciones),
                title="MENÚ DE ACCIONES",
                border_style="cyan",
            )
        )

        console.print("[bold cyan]Opciones:[/]")
        console.print("  1) Añadir acción")
        console.print("  2) Eliminar acción")
        console.print("  3) Guardar acciones")
        console.print("  4) Cargar acciones")
        console.print("  5) Establecer acciones actuales como predeterminadas")
        console.print("  6) Volver al menú principal")
        
        choice = Prompt.ask("Seleccione una opción")

        match choice:
            case "1":
                anadir_accion(acciones,console)

            case "2":
                eliminar_accion(acciones,console)

            case "3":
                guardar_acciones_interactivo(acciones, console)

            case "4":
                acciones_aux = cargar_acciones(console)
                if acciones_aux is not None:
                    acciones.clear()
                    acciones.extend(acciones_aux)

            case "5":
                set_default_actions(console,acciones)

            case "6":
                console.print("[green]Volviendo al menú principal...[/green]")
                break

            case _ :
                console.print("[red]Opcion no valida (debe introducir un numero entre el 1 y el 6)[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")