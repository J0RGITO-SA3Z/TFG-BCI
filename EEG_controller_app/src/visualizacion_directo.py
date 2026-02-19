from rich.console import Console

from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel

from utils import seleccionarPuertoCOM

import asyncio
import numpy as np
import threading

async def _conectar(console, mgr):
    puerto_COM = seleccionarPuertoCOM(console)

    if puerto_COM == None:
        return None
    
    ok = await mgr.connect(puerto_COM)

    if not ok:
        console.clear()
        console.print(f"[red]Error al conectar EEG[/red]")
        console.input("[dim]Pulse Enter para continuar...[/dim]")
        await mgr.disconnect()
        return None
    else:
        console.clear()
        console.print("[green]EEG conectado correctamente al puerto {puerto_COM}[/green]")
    
    return 0

def _chunk_callback(chunk, chunk_size):
    print(chunk)

def _configurar_EEG(mgr, channelsConfig):
    for ch in channelsConfig:
        mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, ch.enabled)
        mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + ch.index, ch.is_bias)
    
    mgr.set_callback_chunk(_chunk_callback)

async def visualizacion_directo(channelsConfig, console):
    console.clear()
    bacore.init(bacore.Version(2, 0, 0))

    with EEGManager() as mgr:
        if await _conectar(console, mgr) is None:
            bacore.close()
            return
        
        _configurar_EEG(mgr,channelsConfig)

        await mgr.start_stream()
        await asyncio.sleep(3)
        await mgr.stop_stream()
        mgr.disconnect()

    bacore.close()

    return