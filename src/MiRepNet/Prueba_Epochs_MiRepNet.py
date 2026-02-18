
"""
Script para evaluar el modelo MIRepNet con datos EEG personalizados en formato (B, C, T)
con 45 canales de EEG.

"""

import sys
import os, sys, torch, numpy as np
import mne
import torch
import torch.nn.functional as F
from collections import deque
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding

# === Configuración del Dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

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

# Función para convertir un archivo .fif a formato (B, C, T) con 45 canales
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

# === Inicialización del Modelo ===
def load_model(weight_path, device):
    """
    Carga el modelo MIRepNet con los pesos preentrenados.
    
    Args:
        weight_path: Ruta a los pesos del modelo (.pth)
        device: Dispositivo (cuda/cpu)
    
    Returns:
        model: Modelo MIRepNet cargado en eval mode
    """
    # Crear modelo con parámetros estándar
    model = mlm_mask(emb_size=256, depth=6, n_classes=3, pretrainmode=False)
    
    # Configurar embedding para 45 canales
    model.embedding = PatchEmbedding(embed_dim=256, num_channels=45)
    
    # Mover modelo al dispositivo
    model.to(device)
    
    # Cargar pesos preentrenados
    if os.path.isfile(weight_path):
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("✅ Pesos preentrenados cargados correctamente.")
    else:
        print(f"⚠️ No se encontraron pesos en: {weight_path}")
        print("⚠️ El modelo se ejecutará con pesos aleatorios.")
    
    # Pasar a modo evaluación
    model.eval()
    
    return model


def normalize_eeg_data(X, axis=1):
    """
    Normaliza los datos EEG usando z-score normalización por canal.
    
    Args:
        X: Array de datos en formato (B, C, T) o (C, T)
        axis: Eje sobre el cual calcular media y std
    
    Returns:
        X_normalized: Datos normalizados
    """
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized


def predict_batch(model, eeg_data, device, normalize=True):
    """
    Realiza predicciones en un lote de datos EEG.
    
    Args:
        model: Modelo MIRepNet cargado
        eeg_data: Datos EEG en formato (B, C, T) donde C=45
        device: Dispositivo (cuda/cpu)
        normalize: Si normalizar los datos antes de pasar al modelo
    
    Returns:
        predictions: Lista de predicciones (etiquetas)
        probabilities: Tensor con probabilidades [B, n_classes]
        raw_outputs: Tensor de salida bruto del modelo [B, n_classes]
    """
    # Asegurar que es numpy array
    if isinstance(eeg_data, torch.Tensor):
        eeg_data = eeg_data.cpu().numpy()
    
    eeg_data = np.array(eeg_data, dtype=np.float32)
    
    # Validar dimensiones
    if eeg_data.ndim == 2:
        # Si es (C, T), agregar dimensión de batch
        eeg_data = np.expand_dims(eeg_data, axis=0)
    
    if eeg_data.shape[1] != 45:
        raise ValueError(f"Se esperan 45 canales, pero se recibieron {eeg_data.shape[1]}")
    
    B, C, T = eeg_data.shape
    print(f"Datos de entrada: Batch={B}, Canales={C}, Tiempo={T}")
    
    # Normalizar si se especifica
    if normalize:
        eeg_data = normalize_eeg_data(eeg_data, axis=2)  # Normalizar por tiempo
    
    # Convertir a tensor de PyTorch
    X_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
    
    # Forward pass
    with torch.no_grad():
        _, logits = model(X_tensor)  # logits shape: [B, 3]
    
    # Obtener predicciones
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1).cpu().numpy()
    
    return predictions, probabilities, logits


def main():
    """
    Función principal para demostración.
    """
    # === CARGAR MODELO ===
    model = load_model(WEIGHT_PATH, device)
    
    # === DECODIFICADOR DE ETIQUETAS ===
    le = LabelEncoder().fit(["left_hand", "right_hand", "feet"])
    class_names = le.classes_
    
    # === Evaluación de un archivo .fif ===
    archivo = input("Introduce el nombre del archivo a evaluar: ")
    Epochs_x45 = raw_to_epochs(archivo)

    # Realizar predicciones
    predictions, probs, _ = predict_batch(model, Epochs_x45, device, normalize=True)
    
    batch_size = Epochs_x45.shape[0]

    # Mostrar resultados
    print("\n" + "-"*60)
    print("RESULTADOS DE PREDICCIONES:")
    print("-"*60)
    for i in range(batch_size):
        pred_class = le.inverse_transform([predictions[i]])[0]
        confidence = probs[i].max().item() * 100
        print(f"\n Epoch {i}:")
        print(f"   Predicción: {pred_class}")
        print(f"   Confianza: {confidence:.2f}%")
        print(f"   Probabilidades: ", end="")
        for j, class_name in enumerate(class_names):
            print(f"{class_name}={probs[i][j].item():.4f} ", end="")
        print()
    


if __name__ == "__main__":
    main()