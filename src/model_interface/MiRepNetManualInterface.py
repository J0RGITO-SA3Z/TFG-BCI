"""
Implementación de ``ModelInterface`` para el modelo MIRepNet.
"""

import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder

# Asegurar que pretrainedModels está en el path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from MiRepNet.Prueba_Epochs_MiRepNet import LABEL_MAP
from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding
from model_interface.ModelInterface import ModelInterface
from pretrainedModels.MiRepNet.utils.utils import train
from pretrainedModels.MiRepNet.utils.utils import validate

CLASS_NAMES = ["feet", "left_hand", "right_hand"]

class MiRepNetManualInterface(ModelInterface):
    """
    Wrapper de MIRepNet que cumple la interfaz ``ModelInterface``.

    Args:
        weight_path: Ruta al fichero ``.pth`` con los pesos preentrenados.
        device:      ``torch.device`` o cadena (``'cpu'`` / ``'cuda'``).
        emb_size:    Tamaño del embedding del transformer (por defecto 256).
        depth:       Número de bloques transformer (por defecto 6).
        n_classes:   Número de clases de salida (por defecto 3).
        num_channels:Número de canales EEG esperados (por defecto 45).
    """

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def __init__(
        self,
        weight_path: str,
        device: str | torch.device = None,
        emb_size: int = 256,
        depth: int = 6,
        n_classes: int = 3,
        num_channels: int = 45,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.device = device
        self.num_channels = num_channels
        self.n_classes = n_classes

        # Crear modelo con parámetros estándar
        self._model = mlm_mask(emb_size, depth, n_classes, pretrainmode=False)
        
        # Configurar embedding para 45 canales
        self._model.embedding = PatchEmbedding(embed_dim=256, num_channels=45)
        
        # Mover modelo al dispositivo
        self._model.to(device)
        
        # Cargar pesos preentrenados
        if os.path.isfile(weight_path):
            ckpt = torch.load(weight_path, map_location=device)
            self._model.load_state_dict(ckpt, strict=False)
            print("✅ Pesos preentrenados cargados correctamente.")
        else:
            print(f"⚠️ No se encontraron pesos en: {weight_path}")
            print("⚠️ El modelo se ejecutará con pesos aleatorios.")
        
        # Pasar a modo evaluación
        self._model.eval()

        encoder = LabelEncoder()
        self.le = encoder.fit(CLASS_NAMES)

    # ------------------------------------------------------------------
    # Clases auxiliares
    # ------------------------------------------------------------------
    def _getLoader(self, data: np.ndarray, labels: np.ndarray) -> DataLoader:
        X_tensor = torch.tensor(data, dtype=torch.float32)
        y_tensor = torch.tensor(self.le.transform(labels), dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=4
        )
        return loader

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def finetuning(self, trainingData: np.ndarray, trainingLabels: np.ndarray,epochs: int, valData: np.ndarray = None, valLabels: np.ndarray = None):
        """

        Realiza fine-tuning del modelo con los datos de entrenamiento y evalúa con los datos de validación.
        Asume que los datos ya están preprocesados y en formato (B, C, T), y que las etiquetas están codificadas como enteros (0, 1, 2) en el orden de CLASS_NAMES.

            - trainingData: Array numpy con forma (B, C, T) para entrenamiento.    
            - trainingLabels: Array numpy con etiquetas para entrenamiento.
            - valData: Array numpy con forma (B, C, T) para validación.
            - valLabels: Array numpy con etiquetas para validación.
            - epochs: Número de épocas para el fine-tuning.

        Retorna una tupla con la mejor precisión de validación y el historial completo de entrenamiento.

        """
        le = LabelEncoder().fit(CLASS_NAMES)

        y_train = torch.tensor(le.transform(trainingLabels), dtype=torch.long, device=self.device)
        y_val   = torch.tensor(le.transform(valLabels),   dtype=torch.long, device=self.device)

        X_train = torch.tensor(trainingData, dtype=torch.float32, device=self.device)
        X_val   = torch.tensor(valData, dtype=torch.float32, device=self.device)

        # — Optimizador y loss —
        loss_fn = torch.nn.CrossEntropyLoss()

        for name, param in self._model.named_parameters():
            print(name, param.shape)
        # 1. Congelar TODOS los pesos del modelo
        for param in self._model.parameters():
            param.requires_grad = False

        # 2. Descongelar solo la cabeza de clasificación
        # Solo los 2 tensores finales
        self._model.clshead.weight.requires_grad = True
        self._model.clshead.bias.requires_grad   = True

        # 3. Solo actualiza los que tienen requires_grad=True
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=0.001)

        # — Historial para la gráfica —
        history = []  # Lista de diccionarios con keys: train_loss, train_acc, val_loss, val_acc, lr

        print(f"\nIniciando fine-tuning: {len(trainingLabels)} trials train | {len(valLabels)} trials val")
        print("─" * 65)

        for epoch in range(epochs):

            # ── ENTRENAMIENTO ─────────────────────────────────────────────
            self._model.train()
            opt.zero_grad()
            _, out = self._model(X_train)
            loss = loss_fn(out, y_train)
            loss.backward()
            opt.step()

            pred_train = out.argmax(dim=1)
            acc_train  = (pred_train == y_train).float().mean().item()

            # ── VALIDACIÓN (sin gradientes — no queremos actualizar pesos) ─────
            self._model.eval()
            with torch.no_grad():
                _, out_val = self._model(X_val)
                loss_val   = loss_fn(out_val, y_val)
                pred_val   = out_val.argmax(dim=1)
                acc_val    = (pred_val == y_val).float().mean().item()

            current_lr = opt.param_groups[0]['lr']

            # — Guardar historial —
            history.append({
                "train_loss": loss.item(),
                "train_acc": acc_train * 100,
                "val_loss": loss_val.item(),
                "val_acc": acc_val * 100,
                "lr": current_lr
            })


            print(f" Epoch {epoch+1:>3}/{epochs} | "
                f"Train → loss: {loss.item():.4f}  acc: {acc_train*100:.1f}% | "
                f"Val   → loss: {loss_val.item():.4f}  acc: {acc_val*100:.1f}%")

            print("─" * 65)
            best_val_acc = max([h['val_acc'] for h in history])
            print(f"Fine-tuning completado. Mejor val acc: {best_val_acc:.1f}%")


        return (best_val_acc, history)
    
    # Esta parte todavía no la he cambiado para que se ajuste a la nueva clase, pero la dejo aquí para no perderla
    def predict(self, data):
        self._model.eval()
        
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
            _, outputs = self._model(data_tensor)
            
            # Obtener probabilidades con softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            probs_np = probabilities.cpu().numpy()[0]
        
        return probs_np

    def predict_batch(self, data):
        self._model.eval()
        
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            
            _, outputs = self._model(data_tensor)  # cls_output
            
            probabilities = torch.softmax(outputs, dim=1)
            
            probs_np = probabilities.cpu().numpy()
        
        return probs_np

    def __repr__(self) -> str:
        return (
            f"MiRepNetInterface(device={self.device}, "
            f"num_channels={self.num_channels}, n_classes={self.n_classes})"
        )
