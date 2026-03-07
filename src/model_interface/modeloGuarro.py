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
from pretrainedModels.MiRepNet.utils.utils import EA
from pretrainedModels.MiRepNet.utils.utils import pad_missing_channels_diff
from pretrainedModels.MiRepNet.utils.channel_list import use_channels_names

CLASS_NAMES = ["feet", "left_hand", "right_hand"]

def process_and_replace_loader(loader,ischangechn,channels_names):
        all_data = []
        all_labels = []
        for i in range(len(loader.dataset)):
            data, label = loader.dataset[i]
            all_data.append(data.numpy())
            all_labels.append(label)
        
        data_np = np.stack(all_data, axis=0)
        labels_tensor = torch.stack(all_labels)
        
        if ischangechn:
            processed_data = pad_missing_channels_diff(data_np,use_channels_names,channels_names)
            print("after processed：", processed_data.shape)

        processed_data = EA(processed_data).astype(np.float32)  

        new_dataset = TensorDataset(
            torch.from_numpy(processed_data).float(),  
            labels_tensor
        )
        
        loader_args = {
            'batch_size': loader.batch_size,
            'num_workers': loader.num_workers,
            'pin_memory': loader.pin_memory,
            'drop_last': loader.drop_last,
            'shuffle': isinstance(loader.sampler, torch.utils.data.RandomSampler)
        }
        
        return torch.utils.data.DataLoader(new_dataset, **loader_args)

class ModeloGuarro(ModelInterface):
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

            print(
                f"Epoch: {epoch+1}\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, "
                f"LR: {curr_lr:.6f}\n"
            )
        
        return final_val_acc
    
    def _extract_from_epochs(self, data):
        """
        Extrae datos, etiquetas y nombres de canales de un objeto MNE Epochs.

        Returns:
            X             : np.ndarray (B, C, T) con los datos EEG.
            Y             : list[str]  etiquetas textuales por epoch.
            channel_names : list[str]  nombres de los canales EEG.
        """
        channel_names = data.ch_names
        channel_names = [ch.upper() for ch in channel_names]
        X = data.get_data()

        true_labels_numeric = data.events[:, 2]
        inv_event_id = {v: k for k, v in data.event_id.items()}
        Y = [inv_event_id[i] for i in true_labels_numeric]

        return X, Y, channel_names

    def predict(self, data):
        X, Y, channel_names = self._extract_from_epochs(data)

        self._model.eval()
        
        with torch.no_grad():
            data_tensor = torch.from_numpy(X).float().unsqueeze(0).to(self.device)
            _, outputs = self._model(data_tensor)
            
            # Obtener probabilidades con softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            probs_np = probabilities.cpu().numpy()[0]
        
        return probs_np, Y, channel_names

    def epochs_to_dataset(self, data):
        """
        Convierte un objeto MNE Epochs al mismo formato que ``EEGDataset``:
          - X : np.ndarray (B, C, T)  float
          - y : np.ndarray (B,)       int  (etiquetas codificadas con LabelEncoder)

        Útil para pasar directamente a ``train_test_split(X, y, ...)``.
        """
        X, Y_str, channel_names = self._extract_from_epochs(data)
        y = self.le.transform(Y_str)            # list[str] → np.ndarray[int]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y, channel_names

    #Data es un objeto MNE Epochs, se convierte a numpy array (B, C, T) y se predice batch-wise
    def predict_batch(self, data):
        self._model.eval()
        X, y, channel_names = self.epochs_to_dataset(data)
        val_dataset = TensorDataset(X, y)

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4
        )

        val_loader = process_and_replace_loader(
            val_loader, 
            ischangechn=True,
            channels_names=channel_names
        )

        # --- Validación con criterion (reutiliza validate de utils) ---
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = validate(
            self._model, val_loader, criterion, self.device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # --- Recoger probabilidades por batch → (B, 3) ---
        all_probs = []
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(self.device)
                _, outputs = self._model(batch_data)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.append(probabilities.cpu().numpy())

        probs_np = np.concatenate(all_probs, axis=0)   # (epochs, 3)

        return probs_np

    def __repr__(self) -> str:
        return (
            f"MiRepNetInterface(device={self.device}, "
            f"num_channels={self.num_channels}, n_classes={self.n_classes})"
        )
