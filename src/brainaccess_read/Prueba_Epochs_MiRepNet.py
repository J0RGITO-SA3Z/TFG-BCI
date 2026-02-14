import sys
import json
import os, sys, time, torch, numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import deque
from sklearn.preprocessing import LabelEncoder
from set_configuracion import ChannelConfig
from set_configuracion import load_channels_conf
from evaluateRaw import raw_to_epochs

# === Configuración ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "Modelos", "MIRepNet")
WEIGHT_PATH = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(MIREPNET_DIR)
from model.mlm import mlm_mask, PatchEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# === Canales ===
# Cargamos el array de canales desde los archivos JSON en la carpeta Disposition

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(SRC_ROOT, "Disposition", "models")
HEADSET_DIR = os.path.join(CONFIG_DIR, "channels_headset_15.json")


channelsConfig = load_channels_conf(HEADSET_DIR)

# HEADSET_CHANNELS_15 desde Disposition/models
HEADSET_CHANNELS_15 = {ch.index: ch.electrode for ch in channelsConfig if ch.enabled}

print(HEADSET_CHANNELS_15)

# === Modelo MIRepNet ===
# Cargamos el modelo MIRepNet 
# emb_size (tamaño de los vectores de embedding para cada segmento temporal) = 256, depth (profundidad de capas del transformer)=6, n_classes=3 (ajusta n_classes según tu tarea)
model = mlm_mask(emb_size=256, depth=6, n_classes=3)
# embed_dim == embed_size y 45 canales de entrada
model.embedding = PatchEmbedding(embed_dim=256, num_channels=45)
# EXTRA: Selección del dispositivo (CUDA(GPU)/CPU)
model.to(device)

if os.path.isfile(WEIGHT_PATH):
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    print("✅ Pesos preentrenados cargados.")
else:
    print("⚠️ No se encontraron pesos.")

model.eval()

# === Configuración de flujo ===
# Aquí deberías importar tu SDK de BrainAccess:
# from brainaccess import EEGDevice
# deviceEEG = EEGDevice()
# deviceEEG.start_stream()

SAMPLE_RATE = 250        # Hz (ajusta a tu casco)
WINDOW_SEC = 4           # segundos por ventana ->  Típico de Motor Imagery (≈ 1000 muestras)
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC
buffer = deque(maxlen=WINDOW_SIZE)
le = LabelEncoder().fit(["left_hand","right_hand","feet"])


# === Evaluación de un archivo .fif con el modelo MIRepNet ===
archivo = input("Introduce el nombre del archivo a evaluar: ")
Epochs_x45 = raw_to_epochs(archivo)

# Procesamos cada epoch con el modelo MIRepNet
for i in range(Epochs_x45.shape[0]):
    # 1. Extraemos el epoch i
    X = Epochs_x45[i] 

    # 2. Limpieza de dimensiones: quitamos cualquier dimensión de tamaño 1 sobrante
    # Esto asegura que pasamos de algo incierto a (45, T)
    X = np.squeeze(X) 
    
    # 3. Normalización (tu proceso estándar)
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = X - X.mean(axis=1, keepdims=True)

    # 4. Construcción del Tensor 4D: (Batch=1, Extra=1, Canales=45, Tiempo=T)
    # Usamos reshape para no dejar lugar a dudas
    T_samples = X.shape[1]
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(1,45, T_samples).to(device)

    # DEBUG: Descomenta la siguiente línea si vuelve a fallar para ver qué llega al modelo
    # print(f"Shape enviado al modelo: {X_tensor.shape}")

    with torch.no_grad():
        _, logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        label = le.inverse_transform([pred])[0]

    print(f"🔮 Epoch {i} Predicción: {label} Pertenencia: {probabilities}")