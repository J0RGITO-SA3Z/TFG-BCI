import os, sys, torch, numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

# === 1️⃣ Configurar ruta a MIRepNet ===
sys.path.append(r"C:\Users\JORGE\OneDrive\Escritorio\Modelos BCI\Modelos\MIRepNet")
from model.mlm import mlm_mask, PatchEmbedding


# === ⚙️ Selección de dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# === 2️⃣ Cargar dataset Physionet ===
print("📥 Descargando y preparando Physionet Motor Imagery...")
dataset = PhysionetMI()
paradigm = MotorImagery(n_classes=2)
X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])
print(f"✅ Datos cargados: {X.shape}, etiquetas: {np.unique(y)}")

# Normalizar y convertir a tensor
X = (X - X.mean()) / X.std()

# Convertir etiquetas de texto a números
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("🎯 Mapeo de clases:", dict(zip(le.classes_, le.transform(le.classes_))))

# === 5️⃣ Filtrar solo las clases del modelo original ===
mask = np.isin(le.inverse_transform(y), ["left_hand", "right_hand", "feet"])
X = X[mask]
y = y[mask]

le2 = LabelEncoder()
y = le2.fit_transform(le.inverse_transform(y))
print("🎯 Clases seleccionadas:", np.unique(le.inverse_transform(y)))
print("📊 Nuevo tamaño de dataset:", X.shape, y.shape)

# === 🧠 Preparar tensores ===
X = torch.tensor(X, dtype=torch.float32)  # [batch, canales, tiempo]
y = torch.tensor(y, dtype=torch.long)

# Verificar que el eje 1 son los canales
print("Shape antes:", X.shape)  # debería ser [n_trials, n_channels, n_samples]

# Añadir dimensión de “input channel” que usa MIRepNet
#X = X.unsqueeze(1)  # -> [batch, 1, canales, tiempo]
print("Shape tras reshape:", X.shape)

# Mover a dispositivo
X = X.to(device)
y = y.to(device)

print("🧠 Tensores listos:", X.shape, y.shape)

# === 3️⃣ Cargar MIRepNet ===
model = mlm_mask(emb_size=256, depth=6, n_classes=3)
model.embedding = PatchEmbedding(embed_dim=256, num_channels=45)
model.to(device)  # <- importante

print("✅ Modelo MIRepNet inicializado en", device)

# === 3.1️⃣ Cargar pesos preentrenados ===
weight_path = r"C:\Users\JORGE\OneDrive\Escritorio\Modelos BCI\Modelos\MIRepNet\weight\MIRepNet.pth"
try:
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print("✅ Pesos preentrenados cargados.")
except Exception as e:
    print("⚠️ No se pudieron cargar los pesos:", e)

# === 3.2️⃣ Ajustar a 45 canales ===
if X.shape[2] > 45:
    X = X[:, :45, :]
elif X.shape[2] < 45:
    pad = torch.zeros(X.shape[0], 1, 45 - X.shape[2], X.shape[3], device=device)
    X = torch.cat((X, pad), dim=2)

print("🔧 Shape final:", X.shape)

# === 4️⃣ Entrenamiento rápido ===
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 15

for epoch in range(epochs):
    opt.zero_grad()
    _, out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()

    pred = out.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    print(f"🌀 Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")

print("✅ Entrenamiento terminado en", device)
