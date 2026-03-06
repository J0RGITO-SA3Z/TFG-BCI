"""
run_moabb_experiment.py
=======================
Entrena MiRepNet sobre un dataset de MOABB para un sujeto concreto
y visualiza los resultados con PerformanceViewer.

Cambia las variables de la sección CONFIG para probar distintos datasets.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Imports MiRepNet ──────────────────────────────────────────────────────────
from pretrainedModels.MiRepNet.utils.utils import train, validate, process_and_replace_loader
from pretrainedModels.MiRepNet.model.mlm import mlm_mask

# ── Imports visualización ─────────────────────────────────────────────────────
from utils.Performance_Viewer import PerformanceViewer

# ── MOABB ─────────────────────────────────────────────────────────────────────
import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004, BNCI2015_001
from moabb.paradigms import MotorImagery
moabb.set_log_level("ERROR")

# =============================================================================
# CONFIG — cambia aquí para probar distintos datasets/sujetos
# =============================================================================
DATASET_NAME = "BNCI2014001"   # "BNCI2014001" | "BNCI2014004" | "BNCI2015001"
SUBJECT_IDX  = 0               # índice del sujeto (0-based)
EPOCHS       = 20
BATCH_SIZE   = 32
LR           = 1e-3
VAL_SPLIT    = 0.2
SEED         = 42
# =============================================================================

DATASET_MAP = {
    "BNCI2014001": BNCI2014_001,
    "BNCI2014004": BNCI2014_004,
    "BNCI2015001": BNCI2015_001,
}

NUM_CLASSES_MAP = {
    "BNCI2014001": 4,
    "BNCI2014004": 2,
    "BNCI2015001": 2,
}


def load_moabb_data(dataset_name: str, subject_idx: int):
    """Carga datos de MOABB y devuelve (X, y) como numpy arrays."""
    dataset_cls = DATASET_MAP[dataset_name]
    dataset     = dataset_cls()
    subjects    = dataset.subject_list

    subject_id = subjects[subject_idx]
    print(f"Cargando {dataset_name} — sujeto {subject_id} ({subject_idx+1}/{len(subjects)})")

    paradigm = MotorImagery(resample=250.0, fmin=8.0, fmax=30.0)
    X, labels, _ = paradigm.get_data(dataset, subjects=[subject_id])

    # Codificar etiquetas como enteros
    classes   = sorted(set(labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[l] for l in labels], dtype=np.int64)

    print(f"  Shape X: {X.shape}  |  clases: {classes}")
    return X, y, classes


def build_loaders(X, y, dataset_name, val_split=0.2, batch_size=32, seed=42):
    """Divide en train/val y aplica el preprocesado de MiRepNet."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    def to_loader(data, labels, shuffle):
        ds = TensorDataset(
            torch.from_numpy(data).float(),
            torch.from_numpy(labels)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    # Preprocesado MiRepNet: EA + alineación espacial al channel template
    train_loader = process_and_replace_loader(train_loader, ischangechn=True, dataset=dataset_name)
    val_loader   = process_and_replace_loader(val_loader,   ischangechn=True, dataset=dataset_name)

    return train_loader, val_loader


def run(dataset_name, subject_idx, epochs, batch_size, lr, val_split, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Datos
    X, y, classes = load_moabb_data(dataset_name, subject_idx)
    train_loader, val_loader = build_loaders(
        X, y, dataset_name, val_split, batch_size, seed
    )

    # 2. Modelo
    n_classes = NUM_CLASSES_MAP[dataset_name]
    model = mlm_mask(
        emb_size=256,
        depth=6,
        n_classes=n_classes,
        pretrainmode=False,
        pretrain=WEIGHT_PATH
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Bucle de entrenamiento — construimos history para PerformanceViewer
    history = []
    print(f"Entrenando {epochs} épocas...\n")

    for epoch in range(epochs):
        train_loss, train_acc, curr_lr = train(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history.append({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,      # viene en % desde utils
            "val_acc":    val_acc,
            "lr":         curr_lr,
        })

        print(f"  Epoch {epoch+1:>3}/{epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  |  lr={curr_lr:.6f}")

    # 4. Visualización
    viewer = PerformanceViewer()
    viewer.summary(history)
    viewer.plot_fine_tune(history)

    return history


if __name__ == "__main__":
    run(
        dataset_name = DATASET_NAME,
        subject_idx  = SUBJECT_IDX,
        epochs       = EPOCHS,
        batch_size   = BATCH_SIZE,
        lr           = LR,
        val_split    = VAL_SPLIT,
        seed         = SEED,
    )