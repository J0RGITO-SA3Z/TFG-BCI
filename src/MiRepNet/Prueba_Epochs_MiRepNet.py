
"""
Script para evaluar el modelo MIRepNet con datos EEG personalizados en formato (B, C, T)
con 45 canales de EEG.

"""

import sys
import os, sys, torch, numpy as np
import mne
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import deque
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import torch.nn as _nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
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

# Etiquetas del experimento → etiquetas del modelo
LABEL_MAP = {
    "IZQUIERDA" : "left_hand",
    "DERECHA"   : "right_hand",
    "ABAJO"     : "feet",
    "DESCANSO"  : "nothing",
}
CLASS_NAMES = ["feet", "left_hand", "right_hand"]   # orden alfabético = orden real de LabelEncoder
CLASS_NAMES_4 = ["feet", "left_hand", "nothing", "right_hand"]  # 4 clases incluyendo DESCANSO

# Umbral de confianza por defecto: si max(softmax) < CONFIDENCE_THRESHOLD → DESCANSO
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

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
def raw_to_epochs(raw, tmin=0.0, tmax=4.0):
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in LABEL_MAP}
    epochs = mne.Epochs(
        raw,
        events=mne.events_from_annotations(raw)[0],
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True
    )
    epochs = epochs.copy().pick("eeg")

    true_labels_numeric = epochs.events[:, 2]
    inv_event_id        = {v: k for k, v in epochs.event_id.items()}
    true_labels_text    = [inv_event_id[i] for i in true_labels_numeric]

    actual_channels_names = [elem.upper() for elem in epochs.ch_names]
    epochs_data           = epochs.get_data()
    transpolated_data     = pad_missing_channels_diff(epochs_data, use_channels_names, actual_channels_names)

    return transpolated_data, true_labels_text

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

def preprocess_eeg_data(
    raw,
    # ── Filtros ──────────────────────────────────────────
    bandpass        = (8.0, 30.0),   # (l_freq, h_freq) en Hz. None para desactivar
    notch           = None,          # frecuencia notch en Hz (ej. 50.0). None para desactivar
    # ── Resampleo ────────────────────────────────────────
    resample_freq   = 250,           # Hz. None para desactivar
    # ── Rereferencia ─────────────────────────────────────
    apply_car       = True,          # Common Average Reference
    # ── ICA ──────────────────────────────────────────────
    apply_ica       = False,         # ICA para eliminar artefactos
    ica_n_components= 15,            # número de componentes ICA
    ica_method      = "fastica",     # "fastica" | "infomax" | "picard"
):
    """
    Preprocesa un objeto mne.io.Raw aplicando los pasos indicados en orden.

    Pasos (en orden de ejecución):
        1. Bandpass filter
        2. Notch filter
        3. Resampleo
        4. CAR (Common Average Reference)
        5. ICA
        6. Euclidean Alignment

    Args:
        raw             : mne.io.Raw — datos crudos de entrada (no se modifica el original)
        bandpass        : tuple (l_freq, h_freq) o None
        notch           : float o None
        resample_freq   : int o None
        apply_car       : bool
        apply_ica       : bool
        ica_n_components: int
        ica_method      : str
        apply_ea        : bool — normaliza la covarianza entre sujetos (útil para MIRepNet)

    Returns:
        np.ndarray de forma (n_canales, n_muestras) con los datos preprocesados
    """
    print(raw.info['sfreq'])
    # Trabajamos sobre una copia para no modificar el original
    raw = raw.copy()

    # ── 1. Bandpass ──────────────────────────────────────────────────────────
    if bandpass is not None:
        l_freq, h_freq = bandpass
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # ── 2. Notch ─────────────────────────────────────────────────────────────
    if notch is not None:
        raw.notch_filter(freqs=notch, verbose=False)

    # ── 3. Resampleo ─────────────────────────────────────────────────────────
    if resample_freq is not None:
        raw.resample(sfreq=resample_freq, verbose=False)

    # ── 4. CAR ───────────────────────────────────────────────────────────────
    if apply_car:
        raw.set_eeg_reference("average", projection=False, verbose=False)

    # ── 5. ICA ───────────────────────────────────────────────────────────────
    if apply_ica:
        ica = mne.preprocessing.ICA(
            n_components=ica_n_components,
            method=ica_method,
            random_state=42,
            verbose=False,
        )
        ica.fit(raw, verbose=False)
        # Detección automática de artefactos oculares y musculares
        eog_indices, _  = ica.find_bads_eog(raw, verbose=False)  if "eog" in [ch.lower() for ch in raw.ch_names] else ([], None)
        muscle_indices, _ = ica.find_bads_muscle(raw, verbose=False)
        ica.exclude = list(set(eog_indices + muscle_indices))
        raw = ica.apply(raw, verbose=False)

    # ── 6. Extraer datos en numpy ─────────────────────────────────────────────
    data, times = raw[:]   # shape: (n_canales, n_muestras)


    # ── 7. Reconstruir Raw preservando annotations ────────────────────────────
    info = raw.info
    raw_preprocessed = mne.io.RawArray(data, info, verbose=False)
    raw_preprocessed.set_annotations(raw.annotations)

    return raw_preprocessed

def normalize_eeg_data(X):
    """
    Normaliza los datos EEG usando z-score normalización por canal.
    
    Args:
        X: Array de datos en formato (B, C, T) o (C, T)
        axis: Eje sobre el cual calcular media y std
    
    Returns:
        X_normalized: Datos normalizados
    """
    mean = X.mean(axis=(1,2), keepdims=True)
    std = X.std(axis=(1,2), keepdims=True)
    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized

def euclidean_alignment_epochs(X: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment aplicado correctamente sobre epochs (B, C, T).
    Calcula la covarianza media sobre todos los trials y la usa para blanquear
    cada trial individualmente — igual que hace MIRepNet en preentrenamiento.

    Args:
        X: (B, C, T)

    Returns:
        np.ndarray (B, C, T) alineado
    """
    B, C, T = X.shape
    # Covarianza media entre todos los trials
    R_mean = np.mean([X[i] @ X[i].T / T for i in range(B)], axis=0)  # (C, C)
    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # R^{-1/2}
    return np.stack([whitening @ X[i] for i in range(B)], axis=0)  # (B, C, T)

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
        eeg_data = normalize_eeg_data(eeg_data)  # Normalizar por canal y por muestra
    
    # Convertir a tensor de PyTorch
    X_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
    
    # Forward pass
    with torch.no_grad():
        _, logits = model(X_tensor)  # logits shape: [B, 3]
    
    # Obtener predicciones
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1).cpu().numpy()
    
    return predictions, probabilities, logits

def predict_sample(model, eeg_data, device):

    ## Comprobar tipos de entrada ------------------------------------------------------------------------
    if not isinstance(model, _nn.Module):
        raise TypeError("`model` debe ser una instancia de torch.nn.Module")

    if not isinstance(device, torch.device):
        raise TypeError("`device` deve ser una instancia de torch.device")

    if isinstance(eeg_data, torch.Tensor):
        eeg_data = eeg_data.cpu().numpy()

    if not isinstance(eeg_data, np.ndarray):
        raise TypeError("`eeg_data` debe ser un np.ndarray o un torch.Tensor convertible a numpy")

    # eegData debe ser 2D: (C, T)
    if eeg_data.ndim != 2:
        raise ValueError(f"`eeg_data` debe ser 2D (C, T); se recibieron {eeg_data.ndim} dimensiones")

    # Asegurar dtype float32
    eeg_data = np.array(eeg_data, dtype=np.float32)

    # Expandir a batch de tamaño 1 y validar canales
    eeg_batch = np.expand_dims(eeg_data, axis=0)  # (1, C, T)
    if eeg_batch.shape[1] != 45:
        raise ValueError(f"Se esperan 45 canales (C=45), pero se recibieron {eeg_batch.shape[1]}")
    
    ## Uso del modelo -------------------------------------------------------------------------------

    # Convertir a tensor y mover a device
    X_tensor = torch.tensor(eeg_batch, dtype=torch.float32).to(device)

    # Inferencia
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        # soportar modelos que devuelven (aux, logits) o solo logits
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            logits = outputs[1]
        else:
            logits = outputs

    probabilities = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probabilities, dim=1).cpu().numpy()[0])

    return pred_idx, probabilities[0].cpu().numpy(), logits[0].cpu().numpy()

def predict_batch_prueba(model, eeg_data, device):
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
    
    # Convertir a tensor de PyTorch
    X_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
    
    # Forward pass
    with torch.no_grad():
        _, logits = model(X_tensor)  # logits shape: [B, 3]
    
    # Obtener predicciones
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1).cpu().numpy()
    
    return predictions, probabilities, logits

def predict_with_rest_threshold(model, eeg_data, device, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, normalize=True):
    """
    Predice la clase de cada epoch, pero si la probabilidad máxima del modelo
    no supera `confidence_threshold`, clasifica el epoch como 'nothing' (DESCANSO).

    La idea es que el modelo 3-clases (feet / left_hand / right_hand) sólo
    "se compromete" cuando está suficientemente seguro. Cuando no lo está,
    interpretamos esa incertidumbre como ausencia de intención motora → DESCANSO.

    Args:
        model               : Modelo MIRepNet cargado en eval mode.
        eeg_data            : Array (B, C, T) con C=45 canales.
        device              : Dispositivo torch.
        confidence_threshold: float en (0, 1). Por debajo → 'nothing'.
                              Valores típicos: 0.45–0.65.
        normalize           : Si aplicar z-score por epoch antes de inferir.

    Returns:
        pred_labels_4  : list[str] — etiquetas predichas con 4 clases.
        probs          : Tensor [B, 3] — softmax del modelo (siempre 3 clases base).
        max_conf       : np.ndarray [B] — confianza máxima de cada epoch.
        threshold_mask : np.ndarray [B] bool — True donde se aplicó el umbral (→ nothing).
    """
    predictions, probs, _ = predict_batch(model, eeg_data, device, normalize=normalize)
    le = LabelEncoder().fit(CLASS_NAMES)

    max_conf       = probs.max(dim=1).values.cpu().numpy()          # [B]
    threshold_mask = max_conf < confidence_threshold                 # True → reclasificar

    pred_labels_4 = []
    for i, pred_idx in enumerate(predictions):
        if threshold_mask[i]:
            pred_labels_4.append("nothing")
        else:
            pred_labels_4.append(le.inverse_transform([pred_idx])[0])

    return pred_labels_4, probs, max_conf, threshold_mask

##############################################################################
#  FUNCIÓN DOWNSTREAM
##############################################################################

def downstream(archivo=None):
    """
    Evalúa el modelo MIRepNet preentrenado sobre un archivo .fif.
    Al finalizar muestra una gráfica con los resultados.

    Args:
        archivo: Ruta al .fif. Si es None, se pide por consola.
    """
    # — Cargar modelo —
    weight_path = input("Introduce la ruta del archivo de pesos .pth: ").strip()

    if weight_path == "":
        weight_path = WEIGHT_PATH
        print(f"Usando ruta por defecto: {weight_path}")

    model = load_model(weight_path, device)
    le    = LabelEncoder().fit(CLASS_NAMES)

    # — Cargar archivo —
    if archivo is None:
        archivo = input("Introduce la ruta del archivo .fif: ").strip()
    raw = mne.io.read_raw_fif(archivo, preload=True)

    # — Preprocesar —
    raw = preprocess_eeg_data(
        raw,
        bandpass=(8.0, 30.0),
        notch=None,
        resample_freq=250,
        apply_car=True,
        apply_ica=False,
    )

    # — Epochs + etiquetas reales —
    epochs_x45, true_labels_raw = raw_to_epochs(raw)
    epochs_x45 = euclidean_alignment_epochs(epochs_x45)
    # Traducir etiquetas del experimento al formato del modelo
    true_labels = [LABEL_MAP[l] for l in true_labels_raw]

    # — Umbral de confianza para DESCANSO —
    umbral_str = input(f"\nUmbral de confianza para DESCANSO (Enter = {DEFAULT_CONFIDENCE_THRESHOLD}): ").strip()
    try:
        umbral = float(umbral_str) if umbral_str else DEFAULT_CONFIDENCE_THRESHOLD
    except ValueError:
        print(f"⚠️  Valor inválido, usando umbral por defecto: {DEFAULT_CONFIDENCE_THRESHOLD}")
        umbral = DEFAULT_CONFIDENCE_THRESHOLD

    # — Predicciones con umbral —
    pred_labels, probs, max_conf, threshold_mask = predict_with_rest_threshold(
        model, epochs_x45, device, confidence_threshold=umbral, normalize=True
    )

    # — Resumen en consola —
    n        = len(true_labels)
    correct  = [t == p for t, p in zip(true_labels, pred_labels)]
    accuracy = sum(correct) / n * 100

    n_reclasificados = threshold_mask.sum()

    print("\n" + "─" * 70)
    print(f"RESULTADOS DOWNSTREAM  (umbral DESCANSO = {umbral:.2f})")
    print("─" * 70)
    for i in range(n):
        mark   = "✅" if correct[i] else "❌"
        umbral_flag = " [umbral]" if threshold_mask[i] else ""
        print(f" {mark} Epoch {i:>2} | Real: {true_labels[i]:<12} | Pred: {pred_labels[i]:<12} | Conf: {max_conf[i]*100:.1f}%{umbral_flag}")
    print("─" * 70)
    print(f" Epochs reclasificados como DESCANSO por umbral: {n_reclasificados}/{n}")
    print(f" Accuracy total: {sum(correct)}/{n} = {accuracy:.1f}%")
    print("─" * 70)

    # — Gráfica con 4 clases —
    plot_results(true_labels, pred_labels, probs, CLASS_NAMES_4, umbral=umbral, max_conf=max_conf)


##############################################################################
#  FUNCIÓN FINE-TUNE
##############################################################################  
RECORDINGS_DIR = "EEG_controller_app/recordings/"
N_TRAIN_TRIALS = 30   # Igual que en el paper MIRepNet


def _cargar_epochs_sujeto(nombre_sujeto):
    """
    Busca todos los .fif cuyo nombre de archivo contenga `nombre_sujeto`,
    los preprocesa y concatena todos sus epochs en un único array.
    Los epochs con etiqueta DESCANSO/nothing se descartan aquí porque el
    fine-tune solo ajusta la cabeza de 3 clases.

    Args:
        nombre_sujeto : str — ej. "suj1". Búsqueda por substring (case-insensitive).

    Returns:
        X      : np.ndarray (B_total, C=45, T)
        labels : list[str] — etiquetas en formato modelo ("left_hand", etc.)

    Raises:
        FileNotFoundError : Sin .fif para ese sujeto.
        ValueError        : Sin epochs válidos de 3 clases.
    """
    ruta     = Path(RECORDINGS_DIR)
    archivos = sorted(ruta.glob("*.fif"))

    # Coincidencia exacta de nombre de sujeto: "suj2" casa con suj2_1.fif y suj2_3.fif
    # pero NO con suj20_1.fif ni suj21_2.fif
    patron = re.compile(rf"(?i)(^|_){re.escape(nombre_sujeto)}(_|\d*\.fif$)")
    archivos_sujeto = [f for f in archivos if patron.search(f.name)]

    if not archivos_sujeto:
        raise FileNotFoundError(
            f"No se encontró ningún .fif con '{nombre_sujeto}' en '{RECORDINGS_DIR}'.\n"
            f"Archivos disponibles: {[f.name for f in archivos]}"
        )

    print(f"\n📂 Archivos encontrados para '{nombre_sujeto}':")
    for f in archivos_sujeto:
        print(f"   • {f.name}")

    preprocess_cfg = dict(
        bandpass=(8.0, 30.0), notch=None,
        resample_freq=250, apply_car=True, apply_ica=False,
    )

    X_all, labels_all = [], []

    for archivo in archivos_sujeto:
        print(f"\n⏳ Procesando: {archivo.name} ...", end=" ", flush=True)
        raw    = mne.io.read_raw_fif(str(archivo), preload=True, verbose=False)
        raw    = preprocess_eeg_data(raw, **preprocess_cfg)
        X_file, labels_raw = raw_to_epochs(raw)

        # Solo epochs con etiquetas válidas para las 3 clases del modelo
        indices_validos, etiquetas_validas = [], []
        for i, lab in enumerate(labels_raw):
            mapped = LABEL_MAP.get(lab)
            if mapped in CLASS_NAMES:
                indices_validos.append(i)
                etiquetas_validas.append(mapped)

        if not indices_validos:
            print("⚠️  Sin epochs de 3 clases — omitido.")
            continue

        X_all.append(X_file[indices_validos])
        labels_all.extend(etiquetas_validas)
        print(f"✅ {len(indices_validos)} epochs válidos.")

    if not X_all:
        raise ValueError(
            f"Ningún archivo de '{nombre_sujeto}' aportó epochs con etiquetas válidas "
            f"(se esperan: {CLASS_NAMES})."
        )

    X_total = np.concatenate(X_all, axis=0)
    print(f"\n📦 Total epochs para '{nombre_sujeto}': {X_total.shape[0]}")
    return X_total, labels_all

def fine_tune(nombre_sujeto=None, epochs=10, lr=1e-3, save_path=None, n_train=N_TRAIN_TRIALS, seed=42):
    """
    Fine-tunea MIRepNet siguiendo el protocolo del paper:
      • Carga TODOS los .fif que contengan `nombre_sujeto` en RECORDINGS_DIR.
      • Concatena todos sus epochs (solo las 3 clases motoras) y mezcla aleatoriamente.
      • Divide: primeros `n_train` → entrenamiento, el resto → validación.
      • Congela el backbone; solo ajusta la cabeza de clasificación (clshead).

    Args:
        nombre_sujeto : str — ej. "suj1". Si es None se pide por consola.
        epochs        : int — número de épocas de fine-tuning.
        lr            : float — learning rate.
        save_path     : str o None — ruta .pth para guardar pesos (None = preguntar al final).
        n_train       : int — trials de entrenamiento (default: 30, como el paper).
        seed          : int — semilla para reproducibilidad del shuffle.

    Returns:
        model : Modelo fine-tuneado en eval mode.
    """
    # ── Pedir sujeto si no se pasó ────────────────────────────────────────────
    if nombre_sujeto is None:
        nombre_sujeto = input("Nombre del sujeto (ej. suj1, suj2): ").strip()

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    model = load_model(WEIGHT_PATH, device)
    le    = LabelEncoder().fit(CLASS_NAMES)

    # ── Cargar y concatenar todos los epochs del sujeto ───────────────────────
    X_total, labels_total = _cargar_epochs_sujeto(nombre_sujeto)
    B_total = X_total.shape[0]

    if B_total <= n_train:
        raise ValueError(
            f"Solo hay {B_total} epochs para '{nombre_sujeto}', "
            f"pero se necesitan >{n_train} (train={n_train} + al menos 1 de val)."
        )

    # ── Shuffle reproducible y división n_train / resto ───────────────────────
    rng       = np.random.default_rng(seed)
    indices   = rng.permutation(B_total)
    idx_train = indices[:n_train]
    idx_val   = indices[n_train:]

    X_train_raw  = X_total[idx_train]
    X_val_raw    = X_total[idx_val]
    labels_train = [labels_total[i] for i in idx_train]
    labels_val   = [labels_total[i] for i in idx_val]

    print(f"\n📊 División de datos (seed={seed}):")
    print(f"   Train : {len(labels_train)} trials → {dict((c, labels_train.count(c)) for c in CLASS_NAMES)}")
    print(f"   Val   : {len(labels_val)} trials → {dict((c, labels_val.count(c)) for c in CLASS_NAMES)}")

    # ── Euclidean Alignment por separado (como hace MIRepNet) ─────────────────
    X_train_ea = euclidean_alignment_epochs(X_train_raw)
    X_val_ea   = euclidean_alignment_epochs(X_val_raw)

    # ── Convertir a tensores ──────────────────────────────────────────────────
    y_train = torch.tensor(le.transform(labels_train), dtype=torch.long,    device=device)
    y_val   = torch.tensor(le.transform(labels_val),   dtype=torch.long,    device=device)
    X_train = torch.tensor(normalize_eeg_data(X_train_ea), dtype=torch.float32, device=device)
    X_val   = torch.tensor(normalize_eeg_data(X_val_ea),   dtype=torch.float32, device=device)

    # ── Congelar backbone, descongelar solo clshead ───────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    model.clshead.weight.requires_grad = True
    model.clshead.bias.requires_grad   = True

    opt     = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Bucle de fine-tuning ──────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🚀 Iniciando fine-tuning: {n_train} trials train | {len(labels_val)} trials val")
    print("─" * 65)

    for ep in range(epochs):
        # Entrenamiento
        model.train()
        opt.zero_grad()
        _, out = model(X_train)
        loss   = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        acc_train = (out.argmax(dim=1) == y_train).float().mean().item()

        # Validación
        model.eval()
        with torch.no_grad():
            _, out_val = model(X_val)
            loss_val   = loss_fn(out_val, y_val)
            acc_val    = (out_val.argmax(dim=1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc_train * 100)
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val * 100)

        print(f" Epoch {ep+1:>3}/{epochs} | "
              f"Train → loss: {loss.item():.4f}  acc: {acc_train*100:.1f}% | "
              f"Val   → loss: {loss_val.item():.4f}  acc: {acc_val*100:.1f}%")

    print("─" * 65)
    print(f"Fine-tuning completado. Mejor val acc: {max(history['val_acc']):.1f}%")

    # ── Guardar pesos ─────────────────────────────────────────────────────────
    if save_path is None:
        save_path = input("\nRuta para guardar pesos fine-tuneados (Enter para no guardar): ").strip()
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Pesos guardados en {save_path}")

    # ── Gráfica ───────────────────────────────────────────────────────────────
    plot_training(history, epochs)

    return model

def downstream_sim(model,batchs, labels, device):
    ''' 
        Evalúa el modelo MIRepNet preentrenado sobre un batch de datos simulados.
        Al finalizar muestra una gráfica con los resultados.

        Args:
            model   : Modelo MIRepNet cargado.
            batch   : np.ndarray (B, C=45, T) con epochs simulados.
            labels  : list[str] — etiquetas reales en formato modelo.
            device  : Dispositivo torch.

        Returns:
            None (muestra resultados en consola y gráfica)
    '''
    # — Umbral de confianza para DESCANSO —
    umbral_str = input(f"\nUmbral de confianza para DESCANSO (Enter = {DEFAULT_CONFIDENCE_THRESHOLD}): ").strip()
    try:
        umbral = float(umbral_str) if umbral_str else DEFAULT_CONFIDENCE_THRESHOLD
    except ValueError:
        print(f"⚠️  Valor inválido, usando umbral por defecto: {DEFAULT_CONFIDENCE_THRESHOLD}")
        umbral = DEFAULT_CONFIDENCE_THRESHOLD




    # — Predicciones con umbral —
    pred_labels, probs, max_conf, threshold_mask = predict_with_rest_threshold(
        model, batchs, device, confidence_threshold=umbral, normalize=True
    )










    # — Resumen en consola —
    n        = len(labels)
    correct  = [t == p for t, p in zip(labels, pred_labels)]
    accuracy = sum(correct) / n * 100

    n_reclasificados = threshold_mask.sum()

    print("\n" + "─" * 70)
    print(f"RESULTADOS DOWNSTREAM  (umbral DESCANSO = {umbral:.2f})")
    print("─" * 70)
    for i in range(n):
        mark   = "✅" if correct[i] else "❌"
        umbral_flag = " [umbral]" if threshold_mask[i] else ""
        print(f" {mark} Epoch {i:>2} | Real: {true_labels[i]:<12} | Pred: {pred_labels[i]:<12} | Conf: {max_conf[i]*100:.1f}%{umbral_flag}")
    print("─" * 70)
    print(f" Epochs reclasificados como DESCANSO por umbral: {n_reclasificados}/{n}")
    print(f" Accuracy total: {sum(correct)}/{n} = {accuracy:.1f}%")
    print("─" * 70)

    # — Gráfica con 4 clases —
    plot_results(true_labels, pred_labels, probs, CLASS_NAMES_4, umbral=umbral, max_conf=max_conf)


    return
def fine_tune_sim(model, batchsTraining, batchsValidation, labels_train, labels_val, le, device, save_path=None):
    ''' 
        Fine-tunea MIRepNet siguiendo el protocolo del paper:
        Congela el backbone; solo ajusta la cabeza de clasificación (clshead).

        Args:
            model           : Modelo MIRepNet cargado.
            batchsTraining  : np.ndarray (B_train, C=45, T) con epochs de entrenamiento.
            batchsValidation : np.ndarray (B_val, C=45, T) con epochs de validación.
            labels_train    : list[str] — etiquetas de entrenamiento en formato modelo.
            labels_val      : list[str] — etiquetas de validación en formato modelo.
            le              : LabelEncoder ya ajustado con las clases del modelo.
            device          : Dispositivo torch.
            save_path       : str o None — ruta .pth para guardar pesos (None = no guardar).

        Returns:
            model           : Modelo fine-tuneado en eval mode.
    '''

    epochs = 10
    n_train = 30
    # ── Convertir a tensores ──────────────────────────────────────────────────
    y_train = torch.tensor(le.transform(labels_train), dtype=torch.long,    device=device)
    y_val   = torch.tensor(le.transform(labels_val),   dtype=torch.long,    device=device)
    X_train = torch.tensor(normalize_eeg_data(batchsTraining), dtype=torch.float32, device=device)
    X_val   = torch.tensor(normalize_eeg_data(batchsValidation),   dtype=torch.float32, device=device)

    # ── Congelar backbone, descongelar solo clshead ───────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    model.clshead.weight.requires_grad = True
    model.clshead.bias.requires_grad   = True

    lr=1e-3
    opt     = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Bucle de fine-tuning ──────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nIniciando fine-tuning: {n_train} trials train | {len(labels_val)} trials val")
    print("─" * 65)

    for ep in range(epochs):
        # Entrenamiento
        model.train()
        opt.zero_grad()
        _, out = model(X_train)
        loss   = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        acc_train = (out.argmax(dim=1) == y_train).float().mean().item()

        # Validación
        model.eval()
        with torch.no_grad():
            _, out_val = model(X_val)
            loss_val   = loss_fn(out_val, y_val)
            acc_val    = (out_val.argmax(dim=1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc_train * 100)
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val * 100)

        print(f" Epoch {ep+1:>3}/{epochs} | "
              f"Train → loss: {loss.item():.4f}  acc: {acc_train*100:.1f}% | "
              f"Val   → loss: {loss_val.item():.4f}  acc: {acc_val*100:.1f}%")

    print("─" * 65)
    print(f"Fine-tuning completado. Mejor val acc: {max(history['val_acc']):.1f}%")

    # ── Guardar pesos ─────────────────────────────────────────────────────────
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Pesos guardados en {save_path}")

    # ── Gráfica ───────────────────────────────────────────────────────────────
    plot_training(history, epochs)

    return model,history,epochs 

##############################################################################
#  FUNCIONES PARA GRÁFICAR DE RESULTADOS
##############################################################################

def plot_training(history, epochs):
    """
    Grafica la evolución de loss y accuracy (train vs val) por epoch.
    La divergencia entre train y val indica overfitting.
    """
    x = np.arange(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Evolución del Fine-Tuning", fontsize=14, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax1.plot(x, history["train_loss"], color="#4C72B0", marker="o", markersize=4, label="Train")
    ax1.plot(x, history["val_loss"],   color="#DD8452", marker="o", markersize=4, label="Validación")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss por epoch")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(x)

    # Línea en el epoch con menor val loss
    best_loss_epoch = int(np.argmin(history["val_loss"]))
    ax1.axvline(x[best_loss_epoch], color="gray", linestyle="--", linewidth=1,
                label=f"Mejor val (epoch {x[best_loss_epoch]})")
    ax1.legend()

    # ── Accuracy ─────────────────────────────────────────────────────────────
    ax2.plot(x, history["train_acc"], color="#4C72B0", marker="o", markersize=4, label="Train")
    ax2.plot(x, history["val_acc"],   color="#DD8452", marker="o", markersize=4, label="Validación")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy por epoch")
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(x)

    # Línea y anotación en el epoch con mayor val acc
    best_acc_epoch = int(np.argmax(history["val_acc"]))
    ax2.axvline(x[best_acc_epoch], color="gray", linestyle="--", linewidth=1,
                label=f"Mejor val (epoch {x[best_acc_epoch]})")
    ax2.annotate(
        f"{history['val_acc'][best_acc_epoch]:.1f}%",
        xy=(x[best_acc_epoch], history["val_acc"][best_acc_epoch]),
        xytext=(8, -15), textcoords="offset points",
        fontsize=9, color="#DD8452", fontweight="bold"
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_results(true_labels, pred_labels, probs, class_names, umbral=None, max_conf=None):
    """
    Genera tres gráficas en una sola ventana:
      1. Comparación epoch a época (real vs predicho), con barra de confianza opcional
      2. Matriz de confusión simple
      3. Accuracy global por clase

    Soporta tanto 3 clases (predicción pura del modelo) como 4 clases
    cuando se aplica umbral de confianza para DESCANSO ('nothing').

    Args:
        true_labels : list[str] — etiquetas reales
        pred_labels : list[str] — etiquetas predichas (pueden incluir 'nothing')
        probs       : Tensor [B, 3] — probabilidades del modelo base (siempre 3 clases)
        class_names : list[str] — nombres de clases a mostrar (3 o 4 clases)
        umbral      : float o None — umbral de confianza usado (para anotación en gráfica)
        max_conf    : np.ndarray [B] o None — confianza máxima por epoch
    """
    n        = len(true_labels)
    correct  = [t == p for t, p in zip(true_labels, pred_labels)]
    accuracy = sum(correct) / n * 100

    colors_map = {
        "left_hand"  : "#4C72B0",
        "right_hand" : "#DD8452",
        "feet"       : "#55A868",
        "nothing"    : "#8172B2",   # morado para DESCANSO
    }
    bar_colors = [colors_map.get(c, "#999999") for c in class_names]

    # Si hay confianza por epoch → añadimos un subplot extra inferior
    n_rows = 3 if max_conf is not None else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    titulo_umbral = f" — umbral DESCANSO={umbral:.2f}" if umbral is not None else ""
    fig.suptitle(f"Resultados MIRepNet — Downstream Evaluation{titulo_umbral}",
                 fontsize=15, fontweight="bold")

    label_to_idx = {c: i for i, c in enumerate(class_names)}

    # ── Subplot 1: epoch a epoch ─────────────────────────────────────────────
    ax1 = fig.add_subplot(n_rows, 2, (1, 2))

    x = np.arange(n)

    true_idx = [label_to_idx.get(l, 0) for l in true_labels]
    pred_idx = [label_to_idx.get(l, 0) for l in pred_labels]

    for i in range(n):
        color = "#2ecc71" if correct[i] else "#e74c3c"
        ax1.scatter(i, true_idx[i], marker="o", s=120, color=color, zorder=3,
                    edgecolors="white", linewidths=0.8)
        if not correct[i]:
            ax1.annotate(
                pred_labels[i],
                xy=(i, true_idx[i]),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center", fontsize=7, color="#e74c3c",
                arrowprops=dict(arrowstyle="-", color="#e74c3c", lw=0.8)
            )

    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel("Epoch")
    ax1.set_title("Predicción por epoch  (verde=acierto, rojo=fallo — texto indica predicción errónea)")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xticks(x)

    legend_items = [
        mpatches.Patch(color="#2ecc71", label="Acierto"),
        mpatches.Patch(color="#e74c3c", label="Fallo"),
        mpatches.Patch(color="#8172B2", label="Reclasificado como DESCANSO (umbral)"),
    ]
    ax1.legend(handles=legend_items, loc="upper right")

    # ── Subplot 2: matriz de confusión ───────────────────────────────────────
    ax2 = fig.add_subplot(n_rows, 2, 2 * (n_rows - 1) + 1)

    nc = len(class_names)
    conf_matrix = np.zeros((nc, nc), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        ti = label_to_idx.get(t, 0)
        pi = label_to_idx.get(p, 0)
        conf_matrix[ti, pi] += 1

    im = ax2.imshow(conf_matrix, cmap="Blues")
    ax2.set_xticks(range(nc)); ax2.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
    ax2.set_yticks(range(nc)); ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel("Predicción"); ax2.set_ylabel("Real")
    ax2.set_title("Matriz de confusión")

    for i in range(nc):
        for j in range(nc):
            ax2.text(j, i, str(conf_matrix[i, j]),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
                     fontsize=11, fontweight="bold")

    # ── Subplot 3: accuracy por clase ────────────────────────────────────────
    ax3 = fig.add_subplot(n_rows, 2, 2 * (n_rows - 1) + 2)

    acc_per_class = []
    for c in class_names:
        indices = [i for i, t in enumerate(true_labels) if t == c]
        if indices:
            acc_c = sum(correct[i] for i in indices) / len(indices) * 100
        else:
            acc_c = 0.0
        acc_per_class.append(acc_c)

    bars = ax3.bar(class_names, acc_per_class, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax3.axhline(accuracy, color="black", linestyle="--", linewidth=1.5, label=f"Total: {accuracy:.1f}%")
    ax3.set_ylim(0, 110)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Accuracy por clase")
    ax3.legend(fontsize=10)

    for bar, val in zip(bars, acc_per_class):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Subplot 4 (opcional): confianza por epoch ────────────────────────────
    if max_conf is not None and umbral is not None:
        ax4 = fig.add_subplot(n_rows, 2, (3, 4))

        bar_conf_colors = ["#8172B2" if c < umbral else "#4C72B0" for c in max_conf]
        ax4.bar(x, max_conf * 100, color=bar_conf_colors, edgecolor="white", linewidth=0.6)
        ax4.axhline(umbral * 100, color="red", linestyle="--", linewidth=1.5,
                    label=f"Umbral DESCANSO ({umbral*100:.0f}%)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Confianza máxima (%)")
        ax4.set_title("Confianza del modelo por epoch  (morado = reclasificado como DESCANSO)")
        ax4.set_ylim(0, 105)
        ax4.set_xticks(x)
        ax4.legend(fontsize=10)
        ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    #fine_tune(epochs=10,save_path="src/MiRepNet/Pesos/MIRepNet_finetuned4.pth")
    downstream()


if __name__ == "__main__":
    main()

