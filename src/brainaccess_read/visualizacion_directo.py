from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel

import serial.tools.list_ports

import asyncio
import numpy as np
import threading

"""FUNCIONES PARA CONECTAR EEG"""
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

async def visualizacion_directo(channels, console):
    return
        
class EEGVisualizacionDirecto:
    
    def __init__(self, mgr: EEGManager, console: Console):
        self.mgr = mgr
        self.console = console
        return
    
    async def start(self,channelsConfig):
        await self._conectar()
        self._configurar_EEG(channelsConfig)

        await self.mgr.start_stream()
        await asyncio.sleep(3)
        await self.mgr.stop_stream()
        self.mgr.disconnect()

        return
    
    async def _conectar(self):
        self.puerto_COM = seleccionarPuertoCOM(self.console)

        if self.puerto_COM == None:
            return None
        
        ok = await self.mgr.connect(self.puerto_COM)

        if not ok:
            self.console.clear()
            self.console.print(f"[red]Error al conectar EEG[/red]")
            self.console.input("[dim]Pulse Enter para continuar...[/dim]")
            await self.mgr.disconnect()
            return None
        else:
            self.console.clear()
            self.console.print("[green]EEG conectado correctamente al puerto {self.puerto_COM}[/green]")
        
        return 0
    
    def _configurar_EEG(self, channelsConfig):
        for ch in channelsConfig:
            self.mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, ch.enabled)
            self.mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, ch.is_bias)
        
        self.mgr.set_callback_chunk(self._chunk_callback)

    def _chunk_callback(self, chunk, chunk_size):
        self.console.print(chunk)