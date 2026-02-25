"""
Script adaptado para evaluar/fine-tunear MIRepNet con el dataset MOABB BNCI2014001 (3 clases)
Cambios respecto al original:
  - Carga de datos mediante MOABB en vez de .fif propio
  - raw_to_epochs_moabb() reemplaza raw_to_epochs() 
  - LABEL_MAP y CLASS_NAMES actualizados para BNCI2014001
"""

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import mne
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os, sys
from Prueba_Epochs_MiRepNet import (
    pad_missing_channels_diff, preprocess_eeg_data, normalize_eeg_data, plot_training,
    load_model, WEIGHT_PATH, CLASS_NAMES , LABEL_MAP, use_channels_names,device)

# ── Etiquetas BNCI2014001 (3 clases, ignoramos tongue) ──────────────────────
LABEL_MAP = {
    "left_hand"  : "left_hand",
    "right_hand" : "right_hand",
    "feet"       : "feet",
    # "tongue"   : "tongue",  # descomentar si quieres 4 clases
}
CLASS_NAMES = ["feet", "left_hand", "right_hand"]  # orden alfabético = orden LabelEncoder


# ── Carga de datos con MOABB ─────────────────────────────────────────────────
def load_moabb_subject(subject_id=1, session="0train"):
    """
    Descarga y devuelve los datos de un sujeto de BNCI2014001 como mne.Raw.
    
    Args:
        subject_id : int — sujeto (1-9)
        session    : str — "0train" o "1test"
    
    Returns:
        raw : mne.io.Raw con los datos EEG y anotaciones
    """
    dataset = BNCI2014_001()
    # get_data devuelve dict: {subject: {session: {run: raw}}}
    data = dataset.get_data(subjects=[subject_id])
    
    # Concatenar todos los runs de la sesión indicada
    raws = list(data[subject_id][session].values())
    raw = mne.concatenate_raws(raws)
    
    return raw


# ── Reemplaza raw_to_epochs() para datos MOABB ───────────────────────────────
def raw_to_epochs_moabb(raw, tmin=0.0, tmax=4.0):
    """
    Convierte un Raw de MOABB a formato (B, C, T) con 45 canales.
    MOABB usa event_id numérico + stim channel, no annotations de texto.
    
    Returns:
        transpolated_data : np.ndarray (B, 45, T)
        true_labels       : list[str] en formato CLASS_NAMES
    """
    # MOABB pone las clases en events_from_annotations (sí usa annotations internamente)
    events, event_id = mne.events_from_annotations(raw)
    
    # Filtrar solo las clases que nos interesan
    event_id_filtrado = {k: v for k, v in event_id.items() if k in LABEL_MAP}
    
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtrado,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    epochs = epochs.pick("eeg")

    # Etiquetas reales en texto
    inv_event_id     = {v: k for k, v in epochs.event_id.items()}
    true_labels_text = [inv_event_id[i] for i in epochs.events[:, 2]]
    # Mapear al formato del modelo (en este caso son iguales, pero por consistencia)
    true_labels = [LABEL_MAP[l] for l in true_labels_text]

    # Interpolación a 45 canales
    actual_channels_names = [ch.upper() for ch in epochs.ch_names]
    epochs_data           = epochs.get_data()
    transpolated_data     = pad_missing_channels_diff(
                                epochs_data, use_channels_names, actual_channels_names
                            )

    return transpolated_data, true_labels


# ── Fine-tune con MOABB ───────────────────────────────────────────────────────
def fine_tune_single_subject(subject=1, train_ratio=0.30, epochs=10, lr=1e-3, save_path=None):
    """
    Replica el experimento del paper MIRepNet:
    - Train con el 30% de los trials de UN sujeto
    - Val   con el 70% restante del mismo sujeto
    - Mismo sujeto, misma sesión ("0train")
    
    Según el paper, con esto debería alcanzarse ~81.77% en S0 (sujeto 1).
    """
    le    = LabelEncoder().fit(CLASS_NAMES)
    model = load_model(WEIGHT_PATH, device)

    # — Cargar datos del sujeto —
    print(f"\nCargando datos del sujeto {subject}...")
    raw = load_moabb_subject(subject_id=subject, session="0train")

    # — Preprocesar —
    preprocess_cfg = dict(
        bandpass=(8.0, 30.0),
        notch=50.0,
        resample_freq=250,
        apply_car=False,
        apply_ica=False,
        apply_ea=True,
    )
    raw = preprocess_eeg_data(raw, **preprocess_cfg)

    # — Epochs completos del sujeto —
    X_all, labels_all = raw_to_epochs_moabb(raw)
    y_all = np.array(le.transform(labels_all))

    print(f"Total trials: {len(labels_all)} | Clases: {dict(zip(*np.unique(labels_all, return_counts=True)))}")

    # — Split 30% train / 70% val estratificado (misma proporción por clase) —
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train_np, y_val_np = train_test_split(
        X_all, y_all,
        train_size=train_ratio,
        random_state=42,
        stratify=y_all       # misma proporción de cada clase en train y val
    )

    print(f"Train: {len(y_train_np)} trials ({train_ratio*100:.0f}%) | "
          f"Val: {len(y_val_np)} trials ({(1-train_ratio)*100:.0f}%)")

    # — Convertir a tensores —
    y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
    y_val   = torch.tensor(y_val_np,   dtype=torch.long, device=device)

    X_train = torch.tensor(normalize_eeg_data(X_train), dtype=torch.float32, device=device)
    X_val   = torch.tensor(normalize_eeg_data(X_val),   dtype=torch.float32, device=device)

    # — Entrenamiento —
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
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nIniciando fine-tuning: sujeto {subject} | "
          f"{len(y_train_np)} train | {len(y_val_np)} val | {epochs} epochs")
    print("─" * 70)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        opt.zero_grad()
        _, out = model(X_train)
        loss   = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        acc_train = (out.argmax(1) == y_train).float().mean().item()

        # ── Val ──
        model.eval()
        with torch.no_grad():
            _, out_val = model(X_val)
            loss_val   = loss_fn(out_val, y_val)
            acc_val    = (out_val.argmax(1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc_train * 100)
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val * 100)

        if acc_val * 100 > best_val_acc:
            best_val_acc = acc_val * 100
            best_epoch   = epoch + 1

        print(f" Epoch {epoch+1:>3}/{epochs} | "
              f"Train → loss: {loss.item():.4f}  acc: {acc_train*100:.1f}% | "
              f"Val   → loss: {loss_val.item():.4f}  acc: {acc_val*100:.1f}%"
              + (" ◄ best" if acc_val * 100 == best_val_acc else ""))

    print("─" * 70)
    print(f" Mejor val acc: {best_val_acc:.1f}% (epoch {best_epoch})")
    print(f" Paper reporta: 92.41% para S0 con 30% finetuning")
    print("─" * 70)

    # — Guardar —
    if save_path is None:
        save_path = input("\nRuta para guardar pesos (Enter para no guardar): ").strip()
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Pesos guardados en {save_path}")

    plot_training(history, epochs)
    return model


def main():
     fine_tune_single_subject(subject=1, train_ratio=0.30, epochs=30, lr=1e-3)


if __name__ == "__main__":
    main()