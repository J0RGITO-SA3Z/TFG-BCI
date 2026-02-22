from rich import console
import mne
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib

# codigo reutilizada de MIrepNet en la ruta MIrepNet/utils/channel_positions.py
channel_positions = {
    'FP1': (-0.3, 0.9), 'FPZ': (0, 1.0), 'FP2': (0.3, 0.9),
    'AF7': (-0.3, 0.8), 'AF3': (-0.2, 0.8), 'AFZ': (0, 0.8),'AF4': (0.2, 0.8), 'AF8': (0.3, 0.8),
    'F9': (-0.6, 0.7),'F7': (-0.5, 0.7), 'F5': (-0.4, 0.7), 'F3': (-0.3, 0.7), 
    'F1': (-0.15, 0.7), 'FZ': (0, 0.7), 'F2': (0.15, 0.7),
    'F4': (0.3, 0.7), 'F6': (0.4, 0.7), 'F8': (0.5, 0.7), 'F10': (0.6, 0.7),
    'FT9': (-0.7, 0.6),'FT7': (-0.6, 0.6), 'FC5': (-0.5, 0.6), 'FC3': (-0.4, 0.6),
    'FC1': (-0.2, 0.6), 'FCZ': (0, 0.6), 'FC2': (0.2, 0.6),
    'FC4': (0.4, 0.6), 'FC6': (0.5, 0.6), 'FT8': (0.6, 0.6),'FT10': (0.6, 0.7),
    'FTT9': (-1.1, 0.55),'T7': (-1.0, 0.5),'TPP7': (-0.9, 0.45), 'C5': (-0.7, 0.5), 'C3': (-0.4, 0.5),
    'C1': (-0.2, 0.5), 'CZ': (0, 0.5), 'C2': (0.2, 0.5),
    'C4': (0.4, 0.5), 'C6': (0.7, 0.5), 'TPP8': (9.0, 0.45),'T8': (1.0, 0.5),'FTT10': (1.1, 0.55),
    'TP9': (-0.8, 0.4),'TPP9': (-0.7, 0.4),'TP7': (-0.6, 0.4), 'CP5': (-0.5, 0.4), 'CP3': (-0.4, 0.4),
    'CP1': (-0.2, 0.4), 'CPZ': (0, 0.4), 'CP2': (0.2, 0.4),
    'CP4': (0.4, 0.4), 'CP6': (0.5, 0.4), 'TP8': (0.6, 0.4),'TPP10': (0.7, 0.4),'TP10': (0.8, 0.4),
    'P9': (-0.6, 0.3),'P7': (-0.5, 0.3), 'P5': (-0.4, 0.3), 'P3': (-0.3, 0.3),
    'P1': (-0.15, 0.3), 'PZ': (0, 0.3), 'P2': (0.15, 0.3),
    'P4': (0.3, 0.3), 'P6': (0.4, 0.3), 'P8': (0.5, 0.3),'P10': (0.6, 0.3),
    'PO9': (-0.5, 0.2),'PO7': (-0.4, 0.2), 'PO5': (-0.3, 0.2),'PO3': (-0.2, 0.2), 'POZ': (0, 0.2),
    'PO4': (0.2, 0.2), 'PO6': (0.3, 0.2),'PO8': (0.4, 0.2),'PO10': (0.5, 0.2),
    'O1': (-0.2, 0.1), 'OZ': (0, 0.1), 'O2': (0.2, 0.1),
}

use_channels_names = [      
                       #    'FP1', 'FPZ', 'FP2', 
                       #        'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                   #  'PO7',  'PO3', 'POZ',  'PO4', 'PO8', 
                             #  'O1', 'OZ', 'O2',
        ]

def validar_nombre_electrodo(nombre):
    montage = mne.channels.make_standard_montage("standard_1005")
    nombres_mne = montage.ch_names

    nombre = nombre.strip().upper()
    mapa = {ch.upper(): ch for ch in nombres_mne}

    return mapa.get(nombre, None)

# Fucnion reutilizada de MIrepNet en la ruta MIrepNet/utils/utils.py
def pad_missing_channels_diff(x, target_channels, actual_channels):
    B, C, T = x.shape
    num_target = len(target_channels)
    
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])

    target_pos = np.array([channel_positions[ch] for ch in target_channels])
    
    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)  
            weights /= weights.sum()     
            W[i] = weights
    
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]  
    
    return padded


def raw_to_epochs(archivo, tmin=0.0, tmax=4.0):
    raw = mne.io.read_raw_fif(archivo, preload=True)

    events, event_id = mne.events_from_annotations(raw)

    event_id_filtrado = {k: v for k, v in event_id.items() if k in ["IZQUIERDA", "DERECHA", "ABAJO", "DESCANSO"]}
    
    epochs = mne.Epochs(
        raw,
        events=mne.events_from_annotations(raw)[0],
        event_id=event_id_filtrado,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    epochs = epochs.copy().pick("eeg")

    # Códigos numéricos de cada epoch
    true_labels_numeric = epochs.events[:, 2]

    # Mapa inverso: número → nombre
    inv_event_id = {v: k for k, v in epochs.event_id.items()}

    true_labels_text = [inv_event_id[i] for i in true_labels_numeric]

    for i, label in enumerate(true_labels_text):
        print(f"Epoch {i}: {label}")

    actual_channels_names = [ elem.upper() for elem in epochs.ch_names]
    epochs_data = epochs.get_data()
    transpolated_data = pad_missing_channels_diff(epochs_data, use_channels_names, actual_channels_names)

    return transpolated_data   


def __main__():
    archivo = input("Introduce el nombre del archivo a evaluar: ")
    raw = mne.io.read_raw_fif(archivo, preload=True)

    raw.plot(scalings='auto', verbose=False)
    data = raw.get_data()
    for i in range(data.shape[0]):
        plt.figure()
        plt.plot(data[i])
        plt.title(f"Canal {raw.ch_names[i]}")
        plt.xlabel("Tiempo (muestras)")
        plt.ylabel("Amplitud")
        plt.show()

    events, event_id = mne.events_from_annotations(raw)

    event_id_filtrado = {k: v for k, v in event_id.items() if k in ["IZQUIERDA", "DERECHA", "ABAJO", "DESCANSO"]}

    epochs = mne.Epochs(
        raw,
        events=events,              
        event_id=event_id_filtrado,
        tmin=0.0,                 # inicio relativo a la anotacion
        tmax=4.0,                 # duración del epoch
        baseline=None,            # baseline clásico
        preload=True
    )
    
    epochs = epochs.copy().pick("eeg")
    raw.filter(1,40).plot(scalings='auto', verbose=False)

    actual_channels_names = [ elem.upper() for elem in epochs.ch_names]
    epochs_data = epochs.get_data()
    transpolated_data = pad_missing_channels_diff(epochs_data, use_channels_names, actual_channels_names)
    input("Presiona Enter para mostrar los epochs transpolados...")
    """
    info = mne.create_info(
        ch_names=[validar_nombre_electrodo(elem) for elem in use_channels_names],
        sfreq=epochs.info["sfreq"],
        ch_types="eeg"
    )

    epochs_transpolados = mne.EpochsArray(transpolated_data, info, events=epochs.events, event_id=epochs.event_id)

    epochs_transpolados.plot(n_epochs=1, n_channels=45, scalings='auto', title="Epoch Transpolado - 45 canales")
    """
    


if __name__ == "__main__":
    __main__()