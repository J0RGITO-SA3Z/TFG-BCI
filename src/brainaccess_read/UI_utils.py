from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt

import serial.tools.list_ports

def lee_indice(texto):
    try:
        valor = int(texto)
        return valor
    except ValueError:
        return -2


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

def ask_validated(console, prompt_text, validate_fn):
    while True:
        value = Prompt.ask(prompt_text)
        if validate_fn(value):
            return value
        console.print("[red]❌ Valor inválido[/]")