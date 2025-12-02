import os, sys, torch, numpy as np
from sklearn.preprocessing import LabelEncoder

# ========== CONFIG ==========
USE_ROUTE_B_PROJECTION = False   # False = Ruta A (simple). True = Ruta B (proyección 14->45)
WEIGHT_PATH = r"C:\Users\JORGE\OneDrive\Escritorio\Modelos BCI\Modelos\MIRepNet\weight\MIRepNet.pth"
MIREPNET_DIR = r"C:\Users\JORGE\OneDrive\Escritorio\Modelos BCI\Modelos\MIRepNet"

# 1) Tu lista de 14 canales EEG *útiles* (sin REF/BIAS), ordenados como saldrán de tu casco
YOUR_HEADSET_CHANNELS_14 = [
    # <<< Rellena con tus etiquetas de canal: por ejemplo >>>
    # "Fp1","Fp2","F3","F4","C3","Cz","C4","P3","P4","O1","O2","FC3","FC4","CPz"
]

# 2) Plantilla de 45 canales que usó el preentrenado (ejemplo típico BNCI2014-004 / 10-10)
#    Si la tienes explícita en tu fork, usa esa. Aquí un placeholder corto (rellena/ajusta si usas Ruta B)
TEMPLATE_45 = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8',
    'T7', 'T8',
    'TP7', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2'
]

# ========== CARGA MIRepNet ==========
sys.path.append(MIREPNET_DIR)
from model.mlm import mlm_mask, PatchEmbedding
import torch.nn as nn

# Módulo opcional de proyección 14->45
class ChannelProjector(nn.Module):
    """
    Proyección 1x1 para mapear 14 canales a 45: [B, 14, T] -> [B, 45, T]
    Inicialmente mapea canales comunes por identidad aproximada (opcional)
    """
    def __init__(self, in_ch=14, out_ch=45):
        super().__init__()
        # 1x1 conv sobre eje "canales" <-> usamos Conv1d con in_ch->out_ch
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        # inicialización "suave": identidad en los primeros min(in_ch, out_ch)
        with torch.no_grad():
            self.proj.weight.zero_()
            for i in range(min(in_ch, out_ch)):
                self.proj.weight[i, i, 0] = 1.0

    def forward(self, x):  # x: [B, C, T]
        return self.proj(x)

# ========== EJEMPLO: CARGA DE DATOS ==========
# Aquí pongo un "mock" de X,y para que puedas enchufar tu flujo real:
# - Reemplaza esto con la lectura de tu streamer/fichero del casco (devuelve [N, 16, T] por ejemplo).
# - Asegúrate de quitar REF/BIAS y quedarte con los 14 EEG -> [N, 14, T]
N, T = 60, 480
X = np.random.randn(N, 14, T).astype(np.float32)  # <-- remplázalo por tus datos reales
y_text = np.random.choice(["left_hand","right_hand","feet"], size=N)  # etiquetas de ejemplo

# Preprocesado mínimo: normalizar y CAR
X = (X - X.mean()) / (X.std() + 1e-8)
X = X - X.mean(axis=1, keepdims=True)  # CAR simple (resta media por ensayo/canal)

# Codificación de etiquetas a 0..2
le = LabelEncoder()
y = le.fit_transform(y_text)  # feet/left_hand/right_hand -> 0/1/2 (orden depende de fit)
print("Clases:", dict(zip(le.classes_, range(len(le.classes_)))))

# Tensores
X = torch.tensor(X, dtype=torch.float32)   # [N, 14, T]
y = torch.tensor(y, dtype=torch.long)

# ========== RUTA A (simple, 14 canales directos) ==========
if not USE_ROUTE_B_PROJECTION:
    emb_size = 256
    n_classes = 3
    in_channels = 14

    model = mlm_mask(emb_size=emb_size, depth=6, n_classes=n_classes)
    model.embedding = PatchEmbedding(embed_dim=emb_size, num_channels=in_channels)

    # Cargar parcialmente el checkpoint: cogeremos sobre todo el transformer
    if os.path.isfile(WEIGHT_PATH):
        ckpt = torch.load(WEIGHT_PATH, map_constrained='cpu') if hasattr(torch.load, '__call__') else torch.load(WEIGHT_PATH, map_location='cpu')
        model_dict = model.state_dict()
        # Filtra capas compatibles y/o del transformer (independientes del nº de canales)
        pretrained = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        # Extra: si quieres forzar a cargar todo lo del transformer aunque emb_size coincida
        pretrained.update({k: v for k, v in ckpt.items() if ("transformer" in k) and (k in model_dict) and (v.shape == model_dict[k].shape)})
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        print(f"Pesos cargados parcialmente: {len(pretrained)} capas")

    # Entrada esperada: [B, C, T]  (NO pongas dimensión 1 extra, el embedding ya hace unsqueeze)
    assert X.ndim == 3 and X.shape[1] == in_channels, X.shape

# ========== RUTA B (proyección 14->45 para encajar checkpoint completo) ==========
else:
    emb_size = 256
    n_classes = 3
    in_channels = 45

    projector = ChannelProjector(in_ch=14, out_ch=in_channels)
    model = mlm_mask(emb_size=emb_size, depth=6, n_classes=n_classes)
    model.embedding = PatchEmbedding(embed_dim=emb_size, num_channels=in_channels)

    # Proyecta tus 14 canales a 45 antes de entrar al modelo
    def forward_with_projection(x):
        # x: [B, 14, T] -> [B, 45, T]
        x45 = projector(x)
        return model(x45)

    # Carga del checkpoint casi completa (formas coinciden con 45 canales)
    if os.path.isfile(WEIGHT_PATH):
        ckpt = torch.load(WEIGHT_PATH, map_constrained='cpu') if hasattr(torch.load, '__call__') else torch.load(WEIGHT_PATH, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print("Checkpoint cargado (Ruta B).")

# ========== ENTRENAMIENTO RÁPIDO DE PRUEBA ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)
X = X.to(device)
y = y.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 3
for ep in range(epochs):
    opt.zero_grad()
    if USE_ROUTE_B_PROJECTION:
        _, out = forward_with_projection(X)   # incluye proyección 14->45
    else:
        _, out = model(X)                     # 14 directos
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    acc = (out.argmax(1) == y).float().mean().item()
    print(f"Epoch {ep+1}/{epochs} - loss {loss.item():.4f} - acc {acc*100:.1f}%")

# Guardar fine-tune
torch.save(model.state_dict(), "mirepnet_14ch_finetuned.pth")
print("Guardado: mirepnet_14ch_finetuned.pth")
