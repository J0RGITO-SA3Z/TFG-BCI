"""
Script para evaluar el modelo MIRepNet con datos EEG personalizados en formato (B, C, T)
con 45 canales de EEG.

Usa ProcessorPipeline del paquete raw_processing para el preprocesado.
"""

import os
import sys

import numpy as np
import mne
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
sys.path.append(PROJECT_ROOT)

from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding
from raw_processing import (
    RawProcessorPipeline,
    BandpassFilter,
    NotchFilter,
    Resampler,
    CARReference,
    ICAProcessor,
    SpatialInterpolator,
    AnnotationRenamer,
)

# === Configuración del Dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Etiquetas del experimento → etiquetas del modelo
LABEL_MAP = {
    "IZQUIERDA": "left_hand",
    "DERECHA":   "right_hand",
    "ABAJO":     "feet",
}
CLASS_NAMES = ["feet", "left_hand", "right_hand"]  # orden alfabético = orden real de LabelEncoder

# === Pipeline de preprocesamiento ===
# Se aplican TODOS los pasos de preprocesado en orden:
#   1. Bandpass  2. Notch  3. Resampleo  4. CAR  5. ICA
#   6. Renombrado de anotaciones  7. Interpolación espacial a 45 canales
pipeline = RawProcessorPipeline([
    BandpassFilter(8.0, 30.0),
    NotchFilter(50.0),
    Resampler(250),
    CARReference(),
    ICAProcessor(n_components=15),
    AnnotationRenamer(LABEL_MAP),
    SpatialInterpolator(),          # interpola/rellena hasta los 45 canales de MIRepNet
])

# Función para convertir un Raw (ya preprocesado por el pipeline) a epochs (B, C, T)
def raw_to_epochs(raw, tmin=0.0, tmax=4.0):
    """
    Epoquiza un Raw ya preprocesado por el pipeline.
    Las anotaciones ya están renombradas (left_hand, right_hand, feet)
    y el Raw ya tiene 45 canales gracias a SpatialInterpolator.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_id_filtrado = {k: v for k, v in event_id.items() if k in CLASS_NAMES}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )

    true_labels_numeric = epochs.events[:, 2]
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    true_labels = [inv_event_id[i] for i in true_labels_numeric]

    return epochs.get_data(), true_labels

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

    # — Preprocesar con pipeline —
    raw = pipeline.process(raw)

    # — Epochs + etiquetas reales (ya en formato modelo) —
    epochs_x45, true_labels = raw_to_epochs(raw)
    epochs_x45 = euclidean_alignment_epochs(epochs_x45)

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

    # — Preprocesar con pipeline —
    raw_t = pipeline.process(raw_t)
    raw_v = pipeline.process(raw_v)

    # — Epochs + etiquetas (ya en formato modelo) —
    X_train, labels_train = raw_to_epochs(raw_t)
    X_val,   labels_val   = raw_to_epochs(raw_v)

    # — Euclidean Alignment sobre epochs (correcto, por separado train y val) —
    X_train = euclidean_alignment_epochs(X_train)
    X_val   = euclidean_alignment_epochs(X_val)

    y_train = torch.tensor(le.transform(labels_train), dtype=torch.long, device=device)
    y_val   = torch.tensor(le.transform(labels_val),   dtype=torch.long, device=device)

    # Normalizar y convertir a tensor
    X_train = torch.tensor(normalize_eeg_data(X_train), dtype=torch.float32, device=device)
    X_val   = torch.tensor(normalize_eeg_data(X_val),   dtype=torch.float32, device=device)

    # — Optimizador y loss —
    loss_fn = torch.nn.CrossEntropyLoss()

    for name, param in model.named_parameters():
        print(name, param.shape)
    # 1. Congelar TODOS los pesos del modelo
    for param in model.parameters():
        param.requires_grad = False

    # 2. Descongelar solo la cabeza de clasificación
    # Solo los 2 tensores finales
    model.clshead.weight.requires_grad = True
    model.clshead.bias.requires_grad   = True

    # 3. Solo actualiza los que tienen requires_grad=True
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    #opt     = torch.optim.Adam(model.parameters(), lr=lr) # Ajusta pesos de todas las capas (puede ser problemático)

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
    #fine_tune(epochs=10,save_path="src/MiRepNet/Pesos/MIRepNet_finetuned3.pth")
    downstream()


if __name__ == "__main__":
    main()