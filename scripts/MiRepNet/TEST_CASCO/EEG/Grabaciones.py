import sqlite3
import numpy as np
import pandas as pd
import mne
import io

# === 1. Ruta real de tu archivo ===
db_path = r"C:/Users/JORGE/Documents/BrainAccessData/sub-001/ses-001/subj-1_ses-S001_task-_run-001_20251114_164813_eeg.db"

# === 2. Abrir conexión SQLite ===
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# === 3. Leer metadata ===
meta_table = "meta_3148057f-1c6a-42f9-bdef-636769ca705a"
cur.execute(f"SELECT * FROM '{meta_table}';")
channels_str, types_str, units_str, sfreq, uid = cur.fetchone()

ch_names = channels_str.split(",")
ch_types = types_str.split(",")
units = units_str.split(",")

print("Canales:", ch_names)
print("Frecuencia de muestreo:", sfreq)
print("Tipos:", ch_types)

# === 4. Leer filas de datos ===
data_table = "data_3148057f-1c6a-42f9-bdef-636769ca705a"
cur.execute(f"SELECT data, time FROM '{data_table}';")
rows = cur.fetchall()

# === 5. Reconstruir señal ===
all_data = []
all_times = []

for data_blob, time_blob in rows:

    # Cargar blobs .npy desde memoria (ESTA ES LA PARTE CORRECTA)
    arr = np.load(io.BytesIO(data_blob), allow_pickle=False)
    tarr = np.load(io.BytesIO(time_blob), allow_pickle=False)

    all_data.append(arr)
    all_times.append(tarr)

# Convertir a arrays grandes
data = np.vstack(all_data)              # (n_samples, n_channels)
times = np.concatenate(all_times)

print("Forma final:", data.shape)

# === 6. Exportar a CSV ===
df = pd.DataFrame(data, columns=ch_names)
df["timestamp"] = times

csv_path = "brainaccess_export.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")

# === 7. Crear objeto RAW de MNE ===
info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=["eeg" if t == "EEG" else "misc" for t in ch_types]
)

raw = mne.io.RawArray(data.T, info)

fif_path = "brainaccess_raw.fif"
raw.save(fif_path, overwrite=True)
print(f"MNE FIF guardado en: {fif_path}")
