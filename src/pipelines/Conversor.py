import sqlite3
import io
import numpy as np
import pandas as pd
import mne


# ============================================================
#   1) Cargar un archivo .db de BrainAccess → estructura Python
# ============================================================

def load_brainaccess_db(db_path):
    """
    Lee un archivo .db de BrainAccess (formato nuevo) y devuelve
    un diccionario con:
      - data_all: np.array (n_channels, n_times)
      - time_all: np.array (n_times,)
      - meta: dict con canales, sfreq, unidades, etc.
      - annotations_df: dataframe con anotaciones (si existen)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- Detectar tablas ---
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [r[0] for r in cur.fetchall()]

    def get_cols(t):
        cur.execute(f'PRAGMA table_info("{t}");')
        return [c[1] for c in cur.fetchall()]

    data_table = meta_table = ann_table = None

    for t in table_names:
        cols = set(get_cols(t))

        if {"data", "time", "local_clock"} <= cols:
            data_table = t
        elif {"channels", "channels_type", "channels_unit", "sf", "id"} <= cols:
            meta_table = t
        elif {"annotation", "time"} <= cols:
            ann_table = t

    if data_table is None or meta_table is None:
        raise RuntimeError("No se pudieron encontrar tablas data/meta.")

    # --- META ---
    meta_df = pd.read_sql_query(f'SELECT * FROM "{meta_table}";', conn)
    meta_row = meta_df.iloc[0]

    meta = {
        "channels": [c.strip() for c in meta_row["channels"].split(",")],
        "channels_type": [ct.strip() for ct in meta_row["channels_type"].split(",")],
        "channels_unit": [u.strip() for u in meta_row["channels_unit"].split(",")],
        "sfreq": float(meta_row["sf"])
    }

    # --- DATOS ---
    data_df = pd.read_sql_query(
        f'SELECT data, time FROM "{data_table}" ORDER BY local_clock ASC;',
        conn
    )

    data_chunks = []
    time_chunks = []

    for _, row in data_df.iterrows():
        arr = np.load(io.BytesIO(row["data"]))  # (n_channels, n_samples_chunk)
        tarr = np.load(io.BytesIO(row["time"]))  # (n_samples_chunk,)
        data_chunks.append(arr)
        time_chunks.append(tarr)

    conn.close()

    data_all = np.concatenate(data_chunks, axis=1)
    time_all = np.concatenate(time_chunks, axis=0)

    # --- ANNOTATIONS ---
    annotations_df = None
    if ann_table:
        conn2 = sqlite3.connect(db_path)
        annotations_df = pd.read_sql_query(f'SELECT * FROM "{ann_table}";', conn2)
        conn2.close()

    return {
        "data": data_all,
        "time": time_all,
        "meta": meta,
        "annotations": annotations_df
    }


# ============================================================
#   2) Convertir estructura Python → MNE.RawArray
# ============================================================

def to_mne(raw_dict, convert_to_volts=True):
    data = raw_dict["data"]
    time = raw_dict["time"]
    meta = raw_dict["meta"]
    ann_df = raw_dict["annotations"]

    sfreq = meta["sfreq"]
    ch_names = meta["channels"]

    if len(meta["channels_type"]) == len(ch_names):
        ch_types = ["eeg" if ct.lower() == "eeg" else "misc"
                    for ct in meta["channels_type"]]
    else:
        ch_types = ["eeg"] * len(ch_names)

    # Convertir µV → V si procede
    if convert_to_volts and all(u.lower() == "microvolts" for u in meta["channels_unit"]):
        data = data * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Añadir anotaciones
    if ann_df is not None and len(ann_df) > 0:
        onsets_abs = ann_df["time"].to_numpy().astype(float)
        onsets_rel = onsets_abs - time[0]
        durations = np.zeros_like(onsets_rel)
        descriptions = ann_df["annotation"].astype(str).to_numpy()

        raw.set_annotations(mne.Annotations(
            onset=onsets_rel,
            duration=durations,
            description=descriptions
        ))

    return raw


# ============================================================
#   3) Exportar → CSV
# ============================================================

def to_csv(raw_dict, out_csv):
    data = raw_dict["data"]
    time = raw_dict["time"]
    meta = raw_dict["meta"]

    df = pd.DataFrame(data.T, columns=meta["channels"])
    df.insert(0, "time", time)

    df.to_csv(out_csv, index=False)
    return out_csv


# ============================================================
#   4) Exportar → NumPy .npy
# ============================================================

def to_numpy(raw_dict, out_path):
    np.save(out_path, raw_dict["data"])
    return out_path


# ============================================================
#   5) Exportar → .fif (requiere convertir antes a MNE.RawArray)
# ============================================================

def to_fif(raw, out_fif):
    raw.save(out_fif, overwrite=True)
    return out_fif


db_path = r"C:\Users\JORGE\OneDrive\Documents\GitHub\TFG-BCI\Grabaciones casco\db\subj-1_ses-S001_task-_run-001_20251114_164813_eeg.db"

# 1) Leer .db
raw_dict = load_brainaccess_db(db_path)

# 2) Convertir a MNE
raw = to_mne(raw_dict)

# 4) Exportar formatos
to_csv(raw_dict, "eeg_data.csv")
to_numpy(raw_dict, "eeg_data.npy")
to_fif(raw, "eeg_data_raw.fif")
