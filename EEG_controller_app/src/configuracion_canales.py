from pathlib import Path
import brainaccess.core.eeg_channel as eeg_channel
import json
from rich.table import Table
from rich import box
from datetime import datetime
from rich.console import Console
import mne

from dataclasses import dataclass
from app_utils import CHANNELS_CONF_DIR, DEFAULT_CHANNELS_CONFIG_FILE

@dataclass
class ChannelConfig:
    index: int
    name: str
    enabled: bool
    is_bias: bool
    electrode: str

##########################################################
##                                                      ##
##         FUNCIONES DE VISUALIZACION                   ##
##                                                      ##
##########################################################

## Imprime una tabla con los archivos de configuarción disponibles.
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

## Función auxiliar para leer un número entero del input. Si el texto no es un número válido, devuelve -2 para indicar error.
def read_index(texto):
    try:
        valor = int(texto)
        return valor
    except ValueError:
        return -2

##########################################################
##                                                      ##
##         LÓGICA DEL PROGRAMA                          ##
##                                                      ##
##########################################################


## Verifica que el nombre del electrodo ingresado es valido según los nombres estándar de MNE. 
## Si el problema es de mayúsuclas o espacios, lo corrige automáticamente. Si el nombre no es reconocido, devuelve None.
def validar_nombre_electrodo(nombre):
    montage = mne.channels.make_standard_montage("standard_1005")
    nombres_mne = montage.ch_names

    nombre = nombre.strip().upper()
    mapa = {ch.upper(): ch for ch in nombres_mne}

    return mapa.get(nombre, None)

## Guarda la configuración de los canales EEG en un archivo json.
def save_channels_conf(channels, filename):
    json_path = Path(filename)
    json_path.parent.mkdir(parents=True, exist_ok=True)

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

# Interfaz interactiva para guardar la configuración de los canales EEG.
def save_channels_conf_interactive(console, channels):
    console.clear()
    filename = console.input("\n[cyan]Nombre del archivo para guardar (sin extensión)[/cyan]: ").strip()

    if filename == "":
        console.print("[yellow]Guardado cancelado[/yellow]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return

    json_path = Path(CHANNELS_CONF_DIR) / f"{filename}.json"

    save_channels_conf(channels, json_path)

    console.print(f"[green]Configuración guardada en {json_path}[/green]")
    console.input("[dim]Pulse Enter para continuar...[/dim]")

# Funcion que carga la configuración de los canales de un archivo json. 
# Si el archivo contiene nombres de electrodos no reconocidos por MNE, devuelve None para indicar error. 
# En caso contrario, devuelve la lista de ChannelConfig cargada.
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
            electrode= validar_nombre_electrodo(ch["name"]) 
        )
        channels.append(channel)

        if channel.electrode is None:
            return None 
    return channels

## Devuelve una lista de los archivos de configuración disponibles.
def list_json_configs(data_dir: Path):
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.json"))

## Menú para seleccionar un archivo de configuración.
def choose_config_interactive(console, channels):
    console.clear()
    files = list_json_configs(Path(CHANNELS_CONF_DIR))

    if not files:
        console.print("No hay JSONs en la carpeta configs/")
        return None
    
    print_configs_table(Console(), files)

    indice = -2

    while indice < -1 or indice >= len(files):
        s = console.input("\nElige índice (ENTER para cancelar): ").strip()
        if s == "":
            indice = -1
        elif s.isdigit():
            idx = int(s)
            if 0 <= idx < len(files):
                indice = idx
        else:
            console.print("Índice inválido. Prueba otra vez.")

    if indice == -1:
        console.print("[yellow]Carga cancelada[/yellow]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return
    
    archivo = files[idx]

    loaded_channels = load_channels_conf(archivo)

    if loaded_channels is None:
        console.print(f"[red]Error: El archivo contiene nombres de electrodos no válidos para MNE.[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        return
    
    channels.clear()
    channels.extend(loaded_channels)
    console.print(f"[green]Configuración cargada desde {archivo.name}[/green]")
    console.input("[dim]Pulse Enter para continuar...[/dim]")

## Carga la configuracion por defecto para los canales EEG
def load_default_channel_config(channels):
    if not DEFAULT_CHANNELS_CONFIG_FILE.exists():
        return -1
    
    loaded_channels = load_channels_conf(DEFAULT_CHANNELS_CONFIG_FILE)

    if loaded_channels is None:
        return -1
    
    channels.clear()
    channels.extend(loaded_channels)

    return 0

def set_default_config(console, channels):
    console.clear()
    save_channels_conf(channels, DEFAULT_CHANNELS_CONFIG_FILE)
    console.print(f"[green]La configuración actual se ha establecido como configuración por defecto.[/green]")
    console.input("[green]Esto implica que al iniciar el programa se cargará automáticamente esta configuración.[/]")
    console.input("[dim]Pulse Enter para continuar...[/dim]")

## Función que permite activar o desactivar un canal. El usuario ingresa el índice del canal a modificar. 
def toggle_channel(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = read_index(console.input("\n[cyan]Índice del canal a activar/desactivar[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = read_index(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))

    if (idx != -1):
        channels[idx].enabled = not channels[idx].enabled

## Función que permite marcar o desmarcar un canal como bias. El usuario ingresa el índice del canal a modificar.
def toggle_bias(channels, console):
    idx = -2
    console.clear()
    console.print(build_channels_table(channels))
    idx = read_index(console.input("\n[cyan]Índice del canal para bias[/cyan] [dim](-1 para cancelar)[/dim]: "))

    while(idx != -1 and ( idx >= len(channels) or idx < -1 )):
        idx = read_index(console.input("[red]Índice inválido. Intente de nuevo:[/red]"))
        
    if (idx != -1):
        channels[idx].is_bias = not channels[idx].is_bias

## Funcion que permite cambiár el nombre de un canal. El usuario ingresa el índice del canal a modificar y el nuevo nombre del electrodo.
def cambiar_electrodo(channels, console):
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
    elec = validar_nombre_electrodo(elec)

    while(elec == None or (elec != "-1" and  len(elec) > 6) ):
        if elec == None:
            elec = console.input("[red]Nombre no válido. Debe ser un nombre de electrodo reconocido por MNE:[/red]")
        else:
            elec = console.input("[red]Nombre inválido. Intente de nuevo:[/red]")

        elec = validar_nombre_electrodo(elec)

    if (elec != "-1"):
        channels[idx].electrode = elec


##########################################################
##                                                      ##
##         MENU DE CONFIGURACIÓN DE CANALES             ##
##                                                      ##
##########################################################

## Construye una tabla que contiene en cada fila un electrodo EEG y su configuracion.
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

## Muestra las opciones disponibles para configurar los canales EEG.
def display_channel_options(console):
    console.print("\n[bold]Opciones:[/bold]")
    console.print("  1) Activar / desactivar canal")
    console.print("  2) Marcar / desmarcar bias")
    console.print("  3) Cambiar electrodo")
    console.print("  4) Cargar configuracion")
    console.print("  5) Guardar configuración")
    console.print("  6) Establecer configuración actual como predeterminada")
    console.print("  7) Menu principal")

## Muestra el menú de configuración de canales EEG completo.
def render_channels_menu(channels, console):
    console.print(build_channels_table(channels))
    display_channel_options(console)

## Función principal del menú de configuración de canales EEG.
def channels_menu(channels, console):
    choice = ""
    while choice != "7":
        console.clear()
        render_channels_menu(channels, console)
        choice = console.input("\n[cyan]Seleccione una opción[cyan]: ")

        match choice:
            case "1":
                toggle_channel(channels, console)

            case "2":
                toggle_bias(channels, console)

            case "3":
                cambiar_electrodo(channels, console)

            case "4":
                choose_config_interactive(console, channels)

            case "5":
                save_channels_conf_interactive(console, channels)
            
            case "6":
                set_default_config(console, channels)

            case "7":
                console.print("[green]Volviendo al menú principal...[/green]")
                break

            case _:
                console.print("[red]Opción no válida[/red]")
                console.input("[dim]Pulse Enter para continuar...[/dim]")
    return