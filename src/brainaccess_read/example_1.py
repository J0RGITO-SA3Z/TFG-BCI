"""
BrainAccess MIDI (core async) -> record EEG -> save .fif

- Conecta por COMx (Windows)
- Activa 15 canales de ELECTRODE_MEASUREMENT
- Graba N segundos
- Exporta a MNE .fif

Requisitos:
  pip install mne numpy
"""

import asyncio
import time
import threading
from collections import deque

import numpy as np
import mne

import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.eeg_manager import EEGManager


# -----------------------------
# Configuración
# -----------------------------
PORT = "COM4"          # <-- CAMBIA ESTO al COM real
SFREQ = 250            # <-- AJUSTA si tu stream va a otra fs
RECORD_SEC = 10        # segundos a grabar

CHANNELS_15 = [
    "F4", "FCZ", "FZ", "FC3", "F3",
    "CZ", "FC4", "C4", "CP4", "P4",
    "C3", "CP3", "PZ", "CPZ", "P3"
]
N_CH = len(CHANNELS_15)


# -----------------------------
# Buffer thread-safe para chunks
# -----------------------------
samples_lock = threading.Lock()
samples_list = []  # iremos guardando arrays (N_CH, chunk_size)


def build_raw_from_samples(samples_chunks: list[np.ndarray], sfreq: float, ch_names: list[str]) -> mne.io.RawArray:
    """
    samples_chunks: lista de arrays shape (N_CH, chunk_size)
    devuelve RawArray con shape final (N_CH, N_TIMES)
    """
    if not samples_chunks:
        raise RuntimeError("No se han recibido muestras. Revisa conexión/puerto/canales.")

    data = np.concatenate(samples_chunks, axis=1)  # (N_CH, total_samples)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# -----------------------------
# Main async
# -----------------------------
async def main():
    # Inicializar core (a veces ya está inicializado en sesiones previas)
    try:
        bacore.init(bacore.Version(2, 0, 0))
    except Exception as e:
        # Si te sale "Already initialized" lo ignoramos
        print(f"⚠️ bacore.init() aviso: {e}")

    print("Core version:", bacore.get_version())

    with EEGManager() as mgr:

        # Callback cuando llega un chunk nuevo
        def chunk_callback(chunk, chunk_size):
            """
            chunk típicamente viene como lista/array con muchos canales.
            Vamos a extraer ELECTRODE_MEASUREMENT + i para i=0..14.
            """
            try:
                # Construimos matriz (N_CH, chunk_size)
                X = np.zeros((N_CH, chunk_size), dtype=np.float32)

                for i in range(N_CH):
                    ch_idx = mgr.get_channel_index(eeg_channel.ELECTRODE_MEASUREMENT + i)
                    # chunk[ch_idx] debería tener chunk_size muestras
                    X[i, :] = np.array(chunk[ch_idx], dtype=np.float32)

                with samples_lock:
                    samples_list.append(X)

            except Exception as ex:
                # No petes el stream por un print
                print(f"⚠️ Error en callback: {ex}")

        # Conectar
        ok = await mgr.connect(PORT)
        if not ok:
            raise RuntimeError(f"No se pudo conectar al dispositivo en {PORT}. Revisa el COM.")

        print(f"✅ Conectado en {PORT}")

        # Set callback
        mgr.set_callback_chunk(chunk_callback)

        # Habilitar los 15 canales EEG (0..14)
        for i in range(N_CH):
            mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + i, True)

        # (Opcional) habilita número de muestra si quieres depurar
        # mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True)

        # Arrancar stream
        await mgr.start_stream()
        print("🟢 Stream iniciado. Grabando...")

        t0 = time.time()
        while time.time() - t0 < RECORD_SEC:
            # Puedes consultar batería/latencia si quieres (no imprescindible)
            # l = await mgr.get_latency()
            # b = await mgr.get_full_battery_info()
            await asyncio.sleep(0.1)

        await mgr.stop_stream()
        mgr.disconnect()
        print("🛑 Stream parado y desconectado.")

    # Construir y guardar FIF
    with samples_lock:
        chunks = list(samples_list)

    raw = build_raw_from_samples(chunks, SFREQ, CHANNELS_15)

    out_fif = time.strftime("%Y%m%d_%H%M%S") + "_brainaccess_midi_15ch_raw.fif"
    raw.save(out_fif, overwrite=True)
    print(f"💾 Guardado: {out_fif}")

    # (Opcional) plot
    raw_f = raw.copy().filter(1, 40, verbose=False)
    raw_f.plot(scalings="auto", verbose=False)


if __name__ == "__main__":
    asyncio.run(main())
    try:
        bacore.close()
    except Exception as e:
        print(f"⚠️ bacore.close() aviso: {e}")

    
