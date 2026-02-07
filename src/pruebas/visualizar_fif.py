import mne
import numpy as np
from pathlib import Path

# === Ruta a tu .fif ===
fif_path = Path("prueba_jorge.fif")

# === 1) Cargar .fif (esto ya devuelve un objeto MNE Raw) ===
raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
raw.plot_sensors(show_names=True)

raw.plot(
    scalings="auto",
    title="EEG – señal en el tiempo",
    show=True
)

# === 2) Imprimir info básica ===
print("\n=== RAW INFO ===")
print(raw)                 # resumen corto
print(raw.info)            # metadata completa (larga)

print("\n=== Canales ===")
print("n_channels:", raw.info["nchan"])
print("ch_names:", raw.ch_names)

print("\n=== Frecuencia de muestreo ===")
print("sfreq:", raw.info["sfreq"], "Hz")

print("\n=== Duración ===")
print("n_times:", raw.n_times)
print("duración:", raw.n_times / raw.info["sfreq"], "s")

# === 3) Extra: imprimir forma de la matriz y primeras muestras ===
data = raw.get_data()      # (n_channels, n_times) en unidades del propio raw (normalmente V en MNE)
print("\n=== DATA ===")
print("shape:", data.shape)
print("primeras 5 muestras del canal 0:", data[0, :5])

# Si quieres ver en microvoltios para debug rápido:
data_uV = data * 1e6
print("primeras 5 muestras del canal 0 (uV):", data_uV[0, :5])

input("pulse Enter para continuar...")
