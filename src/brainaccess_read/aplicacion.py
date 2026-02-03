from unittest import case
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from dataclasses import dataclass
from rich.table import Table
from rich import box
from datetime import datetime

import json
from pathlib import Path

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel
import asyncio
import serial.tools.list_ports

from visualizacion_directo import EEGVisualizacionDirecto
from grabacion import EEGRecorder

"""
Configuración para la lectura de BCI
"""
@dataclass
class ChannelConfig:
    index: int
    name: str
    enabled: bool
    is_bias: bool
    electrode: str

@dataclass
class EEGState:
    connected: bool = False

"""FUNCIONES PARA LA CONFIGURACIÓN DE CANALES EEG"""

# Funciones para trabajar con JSON - cargar y guardar configuraciones
def load_channels_conf(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        electrodes = cfg["electrodes"]
    channels = []
    for ch in electrodes:
        channel = ChannelConfig(
            index=ch["index"],
            name= ch.get("electrode", f"CH{ch['index']+1}"),
            enabled=ch["active"],
            is_bias=ch["bias"],
            electrode= ch["name"]
        )
        channels.append(channel)
    return channels


def save_channels_conf(channels, json_path):
    electrodes = []
    for ch in channels:
        electrode = {
            "name": ch.electrode,
            "index": ch.index,
            "active": ch.enabled,
            "bias": ch.is_bias,
        }
        electrodes.append(electrode)
    cfg = {"electrodes": electrodes}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def list_json_configs(data_dir: Path):
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.json"))


def print_configs_table(console, files):
    table = Table(
        title="Configuraciones disponibles",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        show_lines=False
    )
    table.add_column("Idx", justify="right", style="bold")
    table.add_column("Archivo", style="white")
    table.add_column("Modificado", style="dim")

    for i, p in enumerate(files):

        # Fecha modificación (opcional)
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            mtime = "-"

        table.add_row(str(i), p.name, mtime)

    console.print(table)

def choose_config_interactive(files):
    if not files:
        print("No hay JSONs en la carpeta configs/")
        return None
    
    print_configs_table(Console(), files)

    while True:
        s = input("\nElige índice (ENTER para cancelar): ").strip()
        if s == "":
            return None
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(files):
                return files[idx]
        print("Índice inválido. Prueba otra vez.")

def show_conf_menu(console):
    console.print("\n[bold]Opciones de configuración:[/bold]")
    console.print("  1) Cargar configuración desde archivo")
    console.print("  2) Guardar configuración a archivo")
    console.print("  3) Volver a la configuración de canales")

def conf_menu(channels, console):
    data_dir = Path("src/disposition/configs")
    choice = ""
    new_file = ""

    while choice != "3":
        console.clear()
        console.print(build_channels_table(channels))
        show_conf_menu(console)
        choice = console.input("\n[cyan]Seleccione una opción[cyan]: ")

        match choice:
            case "1":
                files = list_json_configs(data_dir)
                selected_file = choose_config_interactive(files)
                if selected_file:
                    loaded_channels = load_channels_conf(selected_file)
                    channels.clear()
                    channels.extend(loaded_channels)
                    console.print(f"[green]Configuración cargada desde {selected_file.name}[/green]")
                    new_file = selected_file.name
                else:
                    console.print("[yellow]Carga cancelada[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "2":
                filename = console.input("\n[cyan]Nombre del archivo para guardar (sin extensión)[/cyan]: ").strip()
                if filename:
                    save_path = data_dir / f"{filename}.json"
                    save_channels_conf(channels, save_path)
                    console.print(f"[green]Configuración guardada en {save_path.name}[/green]")
                    new_file = filename + ".json"
                else:
                    console.print("[yellow]Guardado cancelado[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "3":
                console.print("[green]Volviendo al menú principal...[/green]")

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
    return new_file

def read_index(texto):
    try:
        valor = int(texto)
        return valor
    except ValueError:
        return -2

def build_channels_table(channels, selected_idx=None):
    table = Table(title="Configuración de canales EEG")

    table.add_column("Idx", justify="right")
    table.add_column("Canal")
    table.add_column("Electrodo")
    table.add_column("Bias")
    table.add_column("Activo")

    for ch in channels:
        is_selected = ch.index == selected_idx
        row_style = "bold on blue" if is_selected else ""

        table.add_row(
            str(ch.index),
            ch.name,
            ch.electrode,
            "[yellow]Sí[/yellow]" if ch.is_bias else "No",
            "[green]Sí[/green]" if ch.enabled else "[red]No[/red]",
            style=row_style
        )

    return table

def show_channels_menu(console):
    console.print("\n[bold]Opciones:[/bold]")
    console.print("  1) Activar / desactivar canal")
    console.print("  2) Marcar / desmarcar bias")
    console.print("  3) Cambiar electrodo")
    console.print("  4) Cambiar archivo de configuración")
    console.print("  5) Menu principal")

def render_channels(channels, console, file):
    console.clear()
    console.print(build_channels_table(channels))
    console.print(f"\n[grey]Archivo de configuración: {file} [grey]")
    show_channels_menu(console)

def toggle_channel(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = read_index(console.input("\n[cyan]Índice del canal a activar/desactivar[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = read_index(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (idx != -1):
        channels[idx].enabled = not channels[idx].enabled

def toggle_bias(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = read_index(console.input("\n[cyan]Índice del canal para bias[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = read_index(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))
        
    if (idx != -1):
        channels[idx].is_bias = not channels[idx].is_bias
        

def change_electrode(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = read_index(console.input("\n[cyan]Índice del canal a cambiar nombre[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = read_index(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (idx == -1):
        return
    
    console.clear()
    console.print(build_channels_table(channels, idx))

    elec = console.input("\n[cyan]nombre para el electrodo con indice "+ str(idx) + " (ej. C3)[/cyan] [dim](máx. 6 caracteres, -1 para cancelar)[/dim]: ")

    while(elec != "-1" and  len(elec) > 6 ):
        elec = console.input("[red]Nombre inválido. Intente de nuevo:[/red]")

    if (elec != "-1"):
        channels[idx].electrode = elec

def channels_menu(channels, console, actual_file):
    choice = ""
    file = actual_file
    while choice != "5":
        render_channels(channels, console, file)
        choice = console.input("\n[cyan]Seleccione una opción[cyan]: ")

        match choice:
            case "1":
                toggle_channel(channels, console)

            case "2":
                toggle_bias(channels, console)

            case "3":
                change_electrode(channels, console)
            case "4":
                file = conf_menu(channels, console)

            case "5":
                console.print("[green]Volviendo al menú principal...[/green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
    return file

"""FUNCIONES DEL MENU PRINCIPAL DE LA APLICACION"""

def build_root_menu(eeg_state):
    # ---- Línea superior: título + estado ----
    title = Text("Menú Principal del Sistema BCI", style="bold")
    subtitle = Text("Entorno de adquisición, configuración y visualización EEG. Introduce el indice de la opcion a ejecutar", style="dim")

    if eeg_state.connected:
        status = Text("EEG CONECTADO", style="bold green")
    else:
        status = Text("EEG DESCONECTADO", style="bold red")

    status.justify = "right"

    header = Columns(
        [title, status],
        expand=True
    )

    # ---- Opciones ----
    menu = Text()

    menu.append("  1) Ver / editar configuración de canales\n")
    menu.append("  2) Test impedancias\n")
    menu.append("  3) Grabar\n")
    menu.append("  4) Tiempo real\n")
    menu.append("  5) Salir\n")

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

async def main_menu(eeg_state,console):

    json_file = "channels_headset_15.json"
    route_json = "src/Disposition/configs/" + json_file
    channels = load_channels_conf(route_json)
    while True:
        console.clear()
        console.print(build_root_menu(eeg_state))

        choice = console.input(">> ")

        match choice:
            case "1":
                json_file = channels_menu(channels, console, json_file)
                console.print("[bold green]Configuración guardada correctamente[/bold green]")

            case "2":
                console.print("[yellow]Test de impedancias (no implementado)[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "3":
                recorder = EEGRecorder(mgr, console)
                recorder.start(channels)

            case "4":
                with EEGManager() as mgr:
                    visualizador = EEGVisualizacionDirecto(mgr, console)
                    await visualizador.start(channels)

            case "5":
                console.print("[bold green]Saliendo del sistema BCI[/bold green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

async def main():
    console = Console()
    eeg_state = EEGState()

    bacore.init(bacore.Version(2, 0, 0))

    await main_menu(eeg_state,console)
    

if __name__ == "__main__":
    asyncio.run(main())