from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from dataclasses import dataclass

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel

import asyncio

import serial.tools.list_ports

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

def lee_indice(texto):
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
    console.print("  4) Menu principal")

def render_channels(channels, console):
    console.clear()
    console.print(build_channels_table(channels))
    show_channels_menu(console)

def toggle_channel(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = lee_indice(console.input("\n[cyan]Índice del canal a activar/desactivar[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = lee_indice(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (idx != -1):
        channels[idx].enabled = not channels[idx].enabled

def toggle_bias(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = lee_indice(console.input("\n[cyan]Índice del canal para bias[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = lee_indice(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))
        
    if (idx != -1):
        channels[idx].is_bias = not channels[idx].is_bias
        

def change_electrode(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = lee_indice(console.input("\n[cyan]Índice del canal a cambiar nombre[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = lee_indice(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (idx == -1):
        return
    
    console.clear()
    console.print(build_channels_table(channels, idx))

    elec = console.input("\n[cyan]nombre para el electrodo con indice "+ str(idx) + " (ej. C3)[/cyan] [dim](máx. 6 caracteres, -1 para cancelar)[/dim]: ")

    while(elec != "-1" and  len(elec) > 6 ):
        elec = console.input("[red]Nombre inválido. Intente de nuevo:[/red]")

    if (elec != "-1"):
        channels[idx].electrode = elec

def channels_menu(channels, console):
    choice = ""

    while choice != "4":
        render_channels(channels, console)
        choice = console.input("\n[cyan]Seleccione una opción[cyan]: ")

        match choice:
            case "1":
                toggle_channel(channels, console)

            case "2":
                toggle_bias(channels, console)

            case "3":
                change_electrode(channels, console)

            case "4":
                console.print("[green]Saliendo de la configuración de canales...[/green]")

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

"""FUNCIONES PARA CONECTAR EEG"""
def seleccionarPuertoCOM(console):
    """
    Función para seleccionar el puerto COM al que está conectado el dispositivo BCI.
    Muestra una tabla con los puertos disponibles y permite al usuario seleccionar uno.
    """

    ports = serial.tools.list_ports.comports()

    table = Table(title="Puertos COM disponibles")
    table.add_column("Índice", justify="right", style="cyan", no_wrap=True)
    table.add_column("Puerto", style="magenta")
    table.add_column("Descripción", style="green")

    for i, port in enumerate(ports):
        table.add_row(str(i), port.device, port.description)

    console.print(table)

    indice = lee_indice(console.input("\nSeleccione puerto COM del dispositivo [dim](-1 para cancelar)[/dim]: "))

    while(indice != -1 and ( indice >= len(ports) or indice < -1 )):
        indice = lee_indice(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (indice == -1):
        return None

    return ports[indice].device

async def conectar_eeg(eeg_state,eeg_mgr,console):
    console.clear()

    if eeg_state.connected:
        eeg_mgr.disconnect()
        eeg_state.connected = False
        console.print("[green]EEG desconectado correctamente[/green]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
    else:
        puerto_COM = seleccionarPuertoCOM(console)

        if puerto_COM == None:
            return
        
        console.print(f"[yellow]Intentando conectar EEG en {puerto_COM}...[/yellow]")

        try:
            ok = await eeg_mgr.connect(puerto_COM)
            if not ok:
                console.clear()
                console.print(f"[red]Error al conectar EEG: {e}[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
                await eeg_mgr.disconnect()
                
            else:
                eeg_state.connected = True
                console.clear()
                console.print("[green]EEG conectado correctamente[/green]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

        except Exception as e:
            console.clear()
            console.print(f"[red]Error al conectar EEG: {e}[/red]")
            console.input("[dim]Pulse Enter para continuar...[/dim]")

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

    if eeg_state.connected:
        menu.append("  1) Desconectar EEG\n")
    else:
        menu.append("  1) Conectar EEG\n")

    menu.append("  2) Ver / editar configuración de canales\n")
    menu.append("  3) Test impedancias\n")
    menu.append("  4) Grabar\n")
    menu.append("  5) Visualizar\n")
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

async def main_menu(eeg_mgr, eeg_state,console):
    channels = [
        ChannelConfig(i, f"CH{i+1}", True, False, "C3")
        for i in range(16)
    ]

    while True:
        console.clear()
        console.print(build_root_menu(eeg_state))

        choice = console.input(">> ")

        match choice:
            case "1":
                await conectar_eeg(eeg_state, eeg_mgr, console)

            case "2":
                channels_menu(channels, console)

            case "3":
                console.print("[yellow]Test de impedancias (no implementado)[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "4":
                console.print("[yellow]Grabación (no implementado)[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "5":
                console.print("[yellow]Visualización (no implementado)[/yellow]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

            case "6":
                console.print("[bold green]Saliendo del sistema BCI[/bold green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")

async def main():
    console = Console()
    eeg_state = EEGState()

    bacore.init(bacore.Version(2, 0, 0))

    with EEGManager() as mgr:
        await main_menu(mgr,eeg_state,console)
    

if __name__ == "__main__":
    asyncio.run(main())