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

from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding
from model_interface.ModelInterface import ModelInterface
from pretrainedModels.MiRepNet.utils.utils import train
from pretrainedModels.MiRepNet.utils.utils import validate

CLASS_NAMES = ["feet", "left_hand", "right_hand"]

class MiRepNetInterface(ModelInterface):
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

        # Crear modelo
        self._model: nn.Module = mlm_mask(
            emb_size=emb_size,
            depth=depth,
            n_classes=n_classes,
            pretrainmode=False,
            pretrain=weight_path
        ).to(self.device)

        self.optimizer = optim.Adam(
            self._model.parameters(), 
            lr=0.001,
            weight_decay=1e-6
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
           self.optimizer, 
            T_max=10
        )

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
        validation = valData is not None and valLabels is not None
        criterion = nn.CrossEntropyLoss()

        train_loader = self._getLoader(trainingData, trainingLabels)

        if validation:
            val_loader = self._getLoader(valData, valLabels)

        history = []
        final_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc, curr_lr = train(
                self._model, train_loader, criterion, 
                self.optimizer, self.device, self.scheduler
            )

            val_loss, val_acc = 0.0, 0.0
            if validation:
                val_loss, val_acc = validate(
                    self._model, val_loader, criterion, self.device
                )
                final_val_acc = val_acc

            history.append({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": curr_lr
            })

            print(
                f"Epoch: {epoch+1}\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"LR: {curr_lr:.6f}\n"
            )
        
        return (final_val_acc, history)
    
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
