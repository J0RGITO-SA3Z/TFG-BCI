from rich.console import Console

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel

from UI_utils import seleccionarPuertoCOM

import asyncio
import numpy as np
import threading

"""FUNCIONES PARA CONECTAR EEG"""
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