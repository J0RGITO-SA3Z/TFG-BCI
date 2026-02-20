
"""
Script para evaluar el modelo MIRepNet con datos EEG personalizados en formato (B, C, T)
con 45 canales de EEG.

"""

import sys
import os, sys, torch, numpy as np
import mne
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

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
}
CLASS_NAMES = ["left_hand", "right_hand", "feet"]

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
    # ── Normalización ────────────────────────────────────
    apply_ea        = False,         # Euclidean Alignment (usado en MIRepNet)
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

    # ── 7. Euclidean Alignment ────────────────────────────────────────────────
    if apply_ea:
        data = _euclidean_alignment(data)
    
    # ── 8. Reconstruir Raw preservando annotations ────────────────────────────
    info = raw.info
    raw_preprocessed = mne.io.RawArray(data, info, verbose=False)
    raw_preprocessed.set_annotations(raw.annotations)

    return raw_preprocessed

def _euclidean_alignment(data: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment (He & Wu, 2019).
    Blanquea los datos para que su covarianza sea la identidad.

    Args:
        data: (n_canales, n_muestras)

    Returns:
        np.ndarray (n_canales, n_muestras) alineado
    """
    cov = np.cov(data)                          # (C, C)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)        # evitar división por cero
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return whitening @ data

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
    model = load_model(WEIGHT_PATH, device)
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
        resample_freq=None,
        apply_car=False,
        apply_ica=True,
        apply_ea=True,
    )

    # — Epochs + etiquetas reales —
    epochs_x45, true_labels_raw = raw_to_epochs(raw)
    # Traducir etiquetas del experimento al formato del modelo
    true_labels = [LABEL_MAP[l] for l in true_labels_raw]

    # — Predicciones —
    predictions, probs, _ = predict_batch(model, epochs_x45, device, normalize=True)
    pred_labels = [le.inverse_transform([p])[0] for p in predictions]

    # — Resumen en consola —
    n        = len(true_labels)
    correct  = [t == p for t, p in zip(true_labels, pred_labels)]
    accuracy = sum(correct) / n * 100

    print("\n" + "─" * 60)
    print("RESULTADOS DOWNSTREAM")
    print("─" * 60)
    for i in range(n):
        mark = "✅" if correct[i] else "❌"
        conf = probs[i].max().item() * 100
        print(f" {mark} Epoch {i:>2} | Real: {true_labels[i]:<12} | Pred: {pred_labels[i]:<12} | Conf: {conf:.1f}%")
    print("─" * 60)
    print(f" Accuracy total: {sum(correct)}/{n} = {accuracy:.1f}%")
    print("─" * 60)

    # — Gráfica —
    plot_results(true_labels, pred_labels, probs, CLASS_NAMES)


##############################################################################
#  FUNCIÓN FINE-TUNE
##############################################################################  
def fine_tune(archivo_train=None, archivo_val=None, epochs=10, lr=1e-3, save_path=None):
    """
    Fine-tunea el modelo MIRepNet con datos propios y grafica la evolución
    de loss y accuracy en cada epoch para detectar overfitting.

    Args:
        archivo_train : Ruta al .fif con los datos de entrenamiento.
        archivo_val   : Ruta al .fif con los datos de validación.
        epochs        : Número de épocas de fine-tuning.
        lr            : Learning rate.
        save_path     : Ruta donde guardar los pesos resultantes (.pth).
                        Si es None se pide por consola al final.
    """
    le = LabelEncoder().fit(CLASS_NAMES)

    # — Cargar modelo —
    model = load_model(WEIGHT_PATH, device)

    # — Cargar archivos —
    if archivo_train is None:
        archivo_train = input("Introduce la ruta del .fif de ENTRENAMIENTO: ").strip()
    if archivo_val is None:
        archivo_val = input("Introduce la ruta del .fif de VALIDACIÓN: ").strip()

    raw_t = mne.io.read_raw_fif(archivo_train, preload=True)
    raw_v = mne.io.read_raw_fif(archivo_val,   preload=True)

    # — Preprocesar (misma config para train y val) —
    preprocess_cfg = dict(
        bandpass=(8.0, 30.0),
        notch=None,
        resample_freq=250,      # MIRepNet espera 250 Hz
        apply_car=True,
        apply_ica=False,
        apply_ea=True,
    )
    raw_t = preprocess_eeg_data(raw_t, **preprocess_cfg)
    raw_v = preprocess_eeg_data(raw_v, **preprocess_cfg)

    # — Epochs + etiquetas —
    X_train, labels_train_raw = raw_to_epochs(raw_t)
    X_val,   labels_val_raw   = raw_to_epochs(raw_v)

    # Traducir etiquetas experimento -> formato modelo -> índice numérico
    labels_train = [LABEL_MAP[l] for l in labels_train_raw]
    labels_val   = [LABEL_MAP[l] for l in labels_val_raw]

    y_train = torch.tensor(le.transform(labels_train), dtype=torch.long, device=device)
    y_val   = torch.tensor(le.transform(labels_val),   dtype=torch.long, device=device)

    # Normalizar y convertir a tensor
    X_train = torch.tensor(normalize_eeg_data(X_train), dtype=torch.float32, device=device)
    X_val   = torch.tensor(normalize_eeg_data(X_val),   dtype=torch.float32, device=device)

    # — Optimizador y loss —
    loss_fn = torch.nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=lr) # Ajusta pesos de todas las capas (puede ser problemático)

    # — Historial para la gráfica —
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    print(f"\nIniciando fine-tuning: {len(labels_train)} trials train | {len(labels_val)} trials val")
    print("─" * 65)

    for epoch in range(epochs):

        # ── ENTRENAMIENTO ─────────────────────────────────────────────
        model.train()
        opt.zero_grad()
        _, out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()

        pred_train = out.argmax(dim=1)
        acc_train  = (pred_train == y_train).float().mean().item()

        # ── VALIDACIÓN (sin gradientes — no queremos actualizar pesos) ─────
        model.eval()
        with torch.no_grad():
            _, out_val = model(X_val)
            loss_val   = loss_fn(out_val, y_val)
            pred_val   = out_val.argmax(dim=1)
            acc_val    = (pred_val == y_val).float().mean().item()

        # — Guardar historial —
        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc_train * 100)
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val * 100)

        print(f" Epoch {epoch+1:>3}/{epochs} | "
              f"Train → loss: {loss.item():.4f}  acc: {acc_train*100:.1f}% | "
              f"Val   → loss: {loss_val.item():.4f}  acc: {acc_val*100:.1f}%")

    print("─" * 65)
    print(f"Fine-tuning completado. Mejor val acc: {max(history['val_acc']):.1f}%")

    # — Guardar pesos —
    if save_path is None:
        save_path = input("\nRuta para guardar pesos fine-tuneados (Enter para no guardar): ").strip()
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Pesos guardados en {save_path}")
    
    # — Gráfica de entrenamiento —
    plot_training(history, epochs) 

    return model

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

def plot_results(true_labels, pred_labels, probs, class_names):
    """
    Genera tres gráficas en una sola ventana:
      1. Comparación epoch a epoch (real vs predicho)
      2. Matriz de confusión simple
      3. Accuracy global con indicador visual
    
    Args:
        true_labels : list[str] — etiquetas reales (en formato modelo, ej. "left_hand")
        pred_labels : list[str] — etiquetas predichas
        probs       : Tensor [B, n_classes] — probabilidades del modelo
        class_names : list[str]
    """
    n        = len(true_labels)
    correct  = [t == p for t, p in zip(true_labels, pred_labels)]
    accuracy = sum(correct) / n * 100

    colors_map = {
        "left_hand"  : "#4C72B0",
        "right_hand" : "#DD8452",
        "feet"       : "#55A868",
    }
    bar_colors = [colors_map.get(c, "#999999") for c in class_names]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Resultados MIRepNet — Downstream Evaluation", fontsize=15, fontweight="bold")

    # ── Subplot 1: epoch a epoch ─────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, (1, 2))   # ocupa las dos columnas superiores

    x = np.arange(n)
    width = 0.35

    # Convertir etiquetas a índices para el eje Y
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    true_idx = [label_to_idx[l] for l in true_labels]
    pred_idx = [label_to_idx[l] for l in pred_labels]

    # Un círculo por epoch: verde si acertó, rojo si falló
    # Si falla, se anota debajo qué predijo el modelo
    for i in range(n):
        color = "#2ecc71" if correct[i] else "#e74c3c"
        ax1.scatter(i, true_idx[i], marker="o", s=120, color=color, zorder=3, edgecolors="white", linewidths=0.8)
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
    ]
    ax1.legend(handles=legend_items, loc="upper right")

    # ── Subplot 2: matriz de confusión ───────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 3)

    nc = len(class_names)
    conf_matrix = np.zeros((nc, nc), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[label_to_idx[t], label_to_idx[p]] += 1

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

    # ── Subplot 3: accuracy global ───────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)

    # Accuracy por clase
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

    plt.tight_layout()
    plt.show()


def main():
    fine_tune()
    #downstream()


if __name__ == "__main__":
    main()