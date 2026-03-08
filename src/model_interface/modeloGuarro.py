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
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIREPNET_DIR  = os.path.join(PROJECT_ROOT, "pretrainedModels", "MIRepNet")
WEIGHT_PATH   = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(MIREPNET_DIR)

# ── Imports visualización ─────────────────────────────────────────────────────
from utils.Performance_Viewer import PerformanceViewer

# ── Imports MiRepNet ──────────────────────────────────────────────────────────
from pretrainedModels.MiRepNet.utils.utils import train, validate, process_and_replace_loader
from pretrainedModels.MiRepNet.model.mlm import mlm_mask

from pretrainedModels.MiRepNet.model.mlm import mlm_mask, PatchEmbedding
from model_interface.ModelInterface import ModelInterface
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

class ModeloGuarro():
    """
    Wrapper de MIRepNet que cumple la interfaz ``ModelInterface``.

    Args:
        weight_path: Ruta al fichero ``.pth`` con los pesos preentrenados.
        device:      ``torch.device`` o cadena (``'cpu'`` / ``'cuda'``).
        emb_size:    Tamaño del embedding del transformer (por defecto 256).
        depth:       Número de bloques transformer (por defecto 6).
        n_classes:   Número de clases de salida (por defecto 3).
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
        num_clases: int = 3,
        channels_names: list[str] = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.channels_names = channels_names
        self.device = device
        self.n_classes = 3

        # Crear modelo
        self._model: nn.Module = mlm_mask(
            emb_size=emb_size,
            depth=depth,
            n_classes=num_clases,
            pretrainmode=False,
            pretrain=weight_path
        ).to(self.device)

    def finetuning_processed(self,X: np.ndarray, Y: np.ndarray,epochs: int):
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=1e-3,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        train_loader = self._build_unprocessed_loader(X, Y, batch_size=32, shuffle=True)

        final_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc, curr_lr = train(
                self._model, train_loader, criterion, 
                self.optimizer, self.device, self.scheduler
            )

            final_val_acc = train_acc

            print(
                f"Epoch: {epoch+1} OK"
                f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
            )
        
        return final_val_acc
    
    def predict_batch_preprocessed(self, X: np.ndarray, batch_size = 32) -> Tuple[np.ndarray, np.ndarray]:
        self._model.eval()
        all_probs = []
        all_preds = []

        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for (data,) in loader:
                data = data.to(self.device)

                _, outputs = self._model(data)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_probs.append(probs.cpu())
                all_preds.append(predicted.cpu())

        probs_array = torch.cat(all_probs).numpy()
        preds_array = torch.cat(all_preds).numpy()

        return preds_array, probs_array

    def finetuning(self,X, y,batch_size=32,seed=42,epochs=20):
        train_loader = self._build_processed_loader(X, y, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=1e-3,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        history = []

        for epoch in range(epochs):
            train_loss, train_acc, curr_lr = train(
                self._model, train_loader, criterion, self.optimizer, self.device, self.scheduler
            )

            history.append({
                "train_loss": train_loss,
                "val_loss":   0,
                "train_acc":  train_acc,      # viene en % desde utils
                "val_acc":    100,
                "lr":         curr_lr,
            })

            print(f"  Epoch {epoch+1:>3}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  |  "
                f"val_loss={0:.4f}  val_acc={100:.1f}%  |  lr={curr_lr:.6f}")
        
        return history
        
    def predict_batch(self, X, batch_size=32) -> Tuple[np.ndarray, np.ndarray]:
        self._model.eval()
        all_probs = []
        all_preds = []

        y = np.zeros(X.shape[0])

        val_loader = self._build_processed_loader(X, y, batch_size=32, shuffle=False)

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                _, outputs = self._model(data)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_probs.append(probs.cpu())
                all_preds.append(predicted.cpu())

        probs_array = torch.cat(all_probs).numpy()
        preds_array = torch.cat(all_preds).numpy()

        return preds_array, probs_array
    
    def validate(self, X, Y):

        val_loader = self._build_processed_loader(X, Y, batch_size=32, shuffle=False)

        _, accuracy, probs_array, preds_array = self._validate_origin(val_loader, None)

        return accuracy, probs_array, preds_array
    
    def _validate_origin(self, val_loader, criterion):
        self._model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                _, outputs = self._model(data)
                _, predicted = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)

                if criterion is not None:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_probs.append(probs.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        accuracy = correct / total * 100

        probs_array = np.concatenate(all_probs, axis=0)
        preds_array = np.concatenate(all_preds, axis=0)

        return epoch_loss, accuracy, probs_array, preds_array
    
    def predict(self, X):
        preds_array, probs_array = self.predict_batch(X)

        return preds_array[0], probs_array[0]

    def experimento(self,X, y, val_split=0.2, batch_size=32, seed=42, epochs=20):
        epoch_predictions = []
        epoch_probabilities = []

        train_loader, val_loader = self._build_loaders(
            X, y, val_split, batch_size, seed
        )

        # 2. Modelo
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=1e-3,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # 3. Bucle de entrenamiento — construimos history para PerformanceViewer
        history = []
        print(f"Entrenando {epochs} épocas...\n")

        for epoch in range(epochs):
            train_loss, train_acc, curr_lr = train(
                self._model, train_loader, criterion, self.optimizer, self.device, self.scheduler
            )
            val_loss, val_acc, probs_array, preds_array = self._validate_origin(val_loader, criterion)

            epoch_predictions.append(preds_array)
            epoch_probabilities.append(probs_array)

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

        return history, epoch_predictions, epoch_probabilities
    
    def _build_processed_loader(self,X, y, batch_size=32, shuffle=True):
        """aplica el preprocesado de MiRepNet."""

        def to_loader(data, labels, shuffle):
            ds = TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(labels)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_loader = to_loader(X, y, shuffle=shuffle)

        # Preprocesado MiRepNet: EA + alineación espacial al channel template
        train_loader = self._process_and_replace_loader(train_loader, ischangechn=True)

        return train_loader
    
    def _build_unprocessed_loader(self,X, Y, batch_size=32, shuffle=False):
        
        def to_loader(data, Y, shuffle, batch_size=32):
            ds = TensorDataset(
            torch.from_numpy(data).float(),
            torch.from_numpy(Y)
            )

            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_loader = to_loader(X, Y, shuffle=shuffle)

        return train_loader
            
    def _build_loaders(self,X, y, val_split=0.2, batch_size=32, seed=42):
        """Divide en train/val y aplica el preprocesado de MiRepNet."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=y
        )

        conteo_Train = Counter(y_train)
        conteo_val = Counter(y_val)

        print(f"Conteo clases Train: {conteo_Train}")
        print(f"Conteo clases Val: {conteo_val}")

        def to_loader(data, labels, shuffle):
            ds = TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(labels)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        train_loader = to_loader(X_train, y_train, shuffle=True)
        val_loader   = to_loader(X_val,   y_val,   shuffle=False)

        # Preprocesado MiRepNet: EA + alineación espacial al channel template
        train_loader = self._process_and_replace_loader(train_loader, ischangechn=True)
        val_loader   = self._process_and_replace_loader(val_loader,   ischangechn=True)

        return train_loader, val_loader
    
    """
    Función reutilizada del repositorio de MiRepNet
    Con modificaciones para aceptar cualquier dataset (canales alineados al template + EA) y devolver un DataLoader nuevo con los datos preprocesados.
    """
    def _process_and_replace_loader(self,loader,ischangechn):
        all_data = []
        all_labels = []
        for i in range(len(loader.dataset)):
            data, label = loader.dataset[i]
            all_data.append(data.numpy())
            all_labels.append(label)
        
        data_np = np.stack(all_data, axis=0)
        labels_tensor = torch.stack(all_labels)
        
        processed_data = EA(data_np).astype(np.float32)  

        if ischangechn and self.channels_names is not None:
            print("before processed：", processed_data.shape)
            channels_names = self.channels_names
            processed_data = pad_missing_channels_diff(processed_data,use_channels_names,channels_names)
            print("after processed：", processed_data.shape)

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
