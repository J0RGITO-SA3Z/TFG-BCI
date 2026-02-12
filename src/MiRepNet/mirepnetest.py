import sys
import json
import os, sys, time, torch, numpy as np
import torch.nn as nn
from collections import deque
from sklearn.preprocessing import LabelEncoder

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
CONFIG_DIR = os.path.join(SRC_ROOT, "Disposition", "configs")
HEADSET_DIR = os.path.join(CONFIG_DIR, "channels_headset_15.json")
TEMPLATE_DIR = os.path.join(CONFIG_DIR, "channels_template_45.json")
headset_file = open (HEADSET_DIR)
template_file = open (TEMPLATE_DIR)

with open(HEADSET_DIR, "r") as f:
    cfg = json.load(f)

HEADSET_CHANNELS_15 = cfg["channels"]

with open(TEMPLATE_DIR, "r") as d:
    cfg = json.load(d)

TEMPLATE_45 = cfg["channels"]

print(HEADSET_CHANNELS_15)
print(TEMPLATE_45)


# ESTA PARTE ES TEMPORAL, HASTA TENER EL INTERPOLADOR ========================
# === Proyector 14→45 ===
class ChannelProjector(nn.Module):
    def __init__(self, in_ch=15, out_ch=45):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.reset_projection()

    def reset_projection(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            for i, ch in enumerate(HEADSET_CHANNELS_15):
                if ch.upper() in [c.upper() for c in TEMPLATE_45]:
                    j = [c.upper() for c in TEMPLATE_45].index(ch.upper())
                    self.proj.weight[j, i, 0] = 1.0

    def forward(self, x):
        return self.proj(x)  # [B,45,T]

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

projector = ChannelProjector().to(device)
model.eval(); projector.eval()

# === Configuración de flujo ===
# Aquí deberías importar tu SDK de BrainAccess:
# from brainaccess import EEGDevice
# deviceEEG = EEGDevice()
# deviceEEG.start_stream()

SAMPLE_RATE = 250        # Hz (ajusta a tu casco)
WINDOW_SEC = 4           # segundos por ventana ->  Típico de Motor Imagery (≈ 480 muestras)
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC
buffer = deque(maxlen=WINDOW_SIZE)

le = LabelEncoder().fit(["left_hand","right_hand","feet"])

# === Bucle en tiempo real ===
print("🧠 Esperando flujo EEG...")
while True:
    # Simulación: datos aleatorios del casco (14 canales, N muestras nuevas)
    # En la práctica: data = deviceEEG.get_data(samples=N)
    new_data = np.random.randn(15, 10).astype(np.float32)  # 10 muestras nuevas

    # Añadir al buffer
    for i in range(new_data.shape[1]):
        buffer.append(new_data[:, i])

    # Solo procesar cuando tenemos ventana completa
    if len(buffer) == WINDOW_SIZE:
        # Convertir buffer a tensor [1,14,T]
        X = np.stack(buffer, axis=1)[None, :, :]  # (1,14,T)
        X = (X - X.mean()) / (X.std() + 1e-8)
        X = X - X.mean(axis=1, keepdims=True)

        X = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            x45 = projector(X)
            _, out = model(x45)
            pred = out.argmax(1).item()
            label = le.inverse_transform([pred])[0]

        print(f"🔮 Predicción: {label}")
        time.sleep(0.25)  # espera simbólica para siguiente lectura