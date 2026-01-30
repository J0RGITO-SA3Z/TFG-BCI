import numpy as np
import threading

class CircularBuffer:
    def __init__(self, n_channels, n_samples):
        self.lock = threading.Lock()

        # Buffer primario (escritura asíncrona)
        self.buffer = np.zeros((n_channels, n_samples))
        self.buffer_idx = 0        # posición de escritura
        self.buffer_end = 0        # cuántas muestras válidas hay

        # Buffer secundario (lectura síncrona)
        self.data = np.zeros((n_channels, n_samples))
        self.data_idx = 0          # posición de escritura
        self.data_end = 0          # cuántas muestras válidas hay

        self.n_samples = n_samples

    def push(self, new_data):
        """
        new_data: shape (n_channels, k)
        """
        k = new_data.shape[1]

        with self.lock:
            end = self.buffer_idx + k

            if end <= self.n_samples:
                # cabe todo seguido
                self.buffer[:, self.buffer_idx:end] = new_data
            else:
                # se parte en dos
                first = self.n_samples - self.buffer_idx
                self.buffer[:, self.buffer_idx:] = new_data[:, :first]
                self.buffer[:, :end % self.n_samples] = new_data[:, first:]
            
            self.buffer_idx = end % self.n_samples
            self.buffer_end = min(self.buffer_end + k, self.n_samples)

    """
    La funcion asume que ya se tiene el lock adquirido.
    Añade al final de data los datos de buffer comprendidos entre inicio y final.
    La funcion asume que los datos en buffer no estan cortados

    No modifica ni principio ni final de data ni de buffer porque espera que lo haga sincroniza()
    """
    def __introducir_data(self, inicio, final):
        assert 0 <= inicio < final <= self.n_samples

        longitud = final - inicio

        if self.data_idx + longitud <= self.n_samples:
            # cabe todo seguido
            self.data[:, self.data_idx:self.data_idx + longitud] = self.buffer[:, inicio:final]
        else:
            # se parte en dos
            first = self.n_samples - self.data_idx
            self.data[:, self.data_idx:] = self.buffer[:, inicio:inicio + first]
            self.data[:, : (longitud - first)] = self.buffer[:, inicio + first:final]
    
    def sincroniza(self):
        corte  = self.buffer_idx - self.buffer_end

        with self.lock:
            if corte >= 0:
                self.__introducir_data(corte, corte + self.buffer_end)
            else:
                # Corte es negativo
                first = self.n_samples + corte
                self.__introducir_data(first, self.n_samples)
                self.__introducir_data( 0, self.buffer_end + corte)
            
            self.data_end = min(self.data_end + self.buffer_end, self.n_samples)
            self.data_idx = (self.data_idx + self.buffer_end) % self.n_samples
            self.buffer_end = 0
            self.buffer_idx = 0

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

def imprimir_menu