import numpy as np
import threading

def chunk_callback(chunk, chunk_size):
        # WARNING: code running inside of callbacks may or may not be running in the reader thread.
        # This means that:
        # - While the callback is running, it might be blocking bluetooth communication
        # - It should be kept as short as possible
        # - It might need a lock/mutex if accessing a shared resource
        # If processing takes too long, getting the main thread's asyncio event loop and using call_soon_threadsafe is advisable.
        print(chunk)
        # for i in range(chunk_size):
        #     print(chunk[mgr.get_channel_index(eeg_channel.ELECTRODE_MEASUREMENT+0)][i])

def imprimir_menu(channels):
    """Aqui configuramos el dispositivo con los datos que llegan en channels, empezamos a escuchar y mostramos los datos en tiempo real con una actualización de 5HZ"""
    """He visto que podemos usar pyqtgraph"""
    return