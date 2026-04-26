"""
Implementación de ``ModelInterface`` para el modelo MIRepNet.
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.preprocessing import LabelEncoder
from typing import Tuple

# ── Rutas ─────────────────────────────────────────────────────────────────────
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MIREPNET_DIR = os.path.join(SRC_ROOT, "components", "pretrainedModels", "MiRepNet")
WEIGHT_PATH  = os.path.join(MIREPNET_DIR, "weight", "MIRepNet.pth")
# MiRepNet's internal files use `from src.*` paths; adding the project
# root lets Python resolve `EEG_app` as a namespace package without modifying
# those files.
PROJ_ROOT = os.path.abspath(os.path.join(SRC_ROOT, "..", ".."))

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MIREPNET_DIR not in sys.path:
    sys.path.append(MIREPNET_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

# ── Imports MiRepNet ──────────────────────────────────────────────────────────
from components.pretrainedModels.MiRepNet.utils.utils import train, validate
from components.pretrainedModels.MiRepNet.model.mlm import mlm_mask

from components.ModelInterface.ModelInterface import ModelInterface

class MiRepNetInterface(ModelInterface):
    """
    Wrapper de MIRepNet que cumple la interfaz ``ModelInterface``.

    Args:
        weight_path: Ruta al fichero ``.pth`` con los pesos preentrenados.
        device:      ``torch.device`` o cadena (``'cpu'`` / ``'cuda'``).
        emb_size:    Tamaño del embedding del transformer (por defecto 256).
        depth:       Número de bloques transformer (por defecto 6).
        training_clases:   Clases que se usarán durante el entrenamiento y que se devolverán al predecir.

    Nota:
    Si se utilizan los pesos preentrenados, el modelo asume que las señales EEG han sido previamente
    procesadas mediante:
        - Euclidean Alignment (EA)
        - Interpolación espacial a un conjunto fijo de canales:

        ['F7','F5','F3','F1','FZ','F2','F4','F6','F8',
         'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
         'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
         'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
         'P7','P5','P3','P1','PZ','P2','P4','P6','P8']

    Esto garantiza la consistencia espacial con los datos usados durante el preentrenamiento.
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
        training_clases: list[str] = ["feet", "left_hand", "right_hand"],
        freeze_encoder = False
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.device = device
        self.training_clases = training_clases
        self._emb_size = emb_size
        self._depth = depth

        # Crear modelo
        self._model: nn.Module = mlm_mask(
            emb_size=emb_size,
            depth=depth,
            n_classes= len(training_clases),
            pretrainmode=False,
        ).to(self.device)

        self._model.init_from_pretrained(weight_path, freeze_encoder=freeze_encoder)

        # Crear codificador de etiquetas (transorma los strings de las clases a números enteros)
        # Los números no tienen porque coincidir exactamente con los del entrenamiento original ya que durante el fine-tuning reaprende
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(training_clases)

    def finetuning(self,X_train: np.ndarray, Y_train: np.ndarray,X_val = None, Y_val = None,epochs = 10):
        # Determina si se tiene que hacer validacion
        validacion = X_val is not None and Y_val is not None

        val_loader = None
        history = []
        
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=1e-3,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Creacion de data loaders.
        # carga datos en batches, permite barajar (shuffle) y mejora la eficiencia del entrenamiento
        Y_train = self.label_encoder.transform(Y_train)
        train_loader = self._build_unprocessed_loader(X_train, Y_train, batch_size=32, shuffle=True)
        
        if validacion:
            Y_val = self.label_encoder.transform(Y_val)
            val_loader = self._build_unprocessed_loader(X_val, Y_val, batch_size=32, shuffle=False)

        #Bucle de entrenamiento
        for epoch in range(epochs):
            val_loss = 0.0
            val_acc = 0.0

            train_loss, train_acc, curr_lr = train(
                self._model, train_loader, criterion, 
                self.optimizer, self.device, self.scheduler
            )

            if validacion:
                val_loss, val_acc = validate(
                    self._model, val_loader, criterion, self.device
                )

            # Guardar métricas en el historial a devolver
            history.append({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,      # viene en % desde utils
                "val_acc":    val_acc, # viene en % desde utils
                "lr":         curr_lr,
            })
            
            # Imprimir métricas del epoch
            print(f"Epoch {epoch+1:>3}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
                  f"val_loss={(val_loss if validacion else 0):.4f} "
                  f"val_acc={(val_acc if validacion else 0):.1f}% | "
                  f"lr={curr_lr:.6f}")
        
        return history
    
    def predict_batch(self, X: np.ndarray, batch_size = 32) -> Tuple[np.ndarray, np.ndarray]:
        self._model.eval()

        # Convertir el array de numpy a un DataLoader de PyTorch
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Iterar sobre el DataLoader y obtener predicciones
        all_probs, all_preds = [], []

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

        # Convertir las probabilidades a un diccionario con las etiquetas de clase y la lista de numeros en lista de etiquetas de clase
        class_labels = self.label_encoder.classes_
        probs_dict = [{label: float(prob) for label, prob in zip(class_labels, probs)} for probs in probs_array]
        preds_array = self.label_encoder.inverse_transform(preds_array)

        return preds_array, probs_dict
    
    def predict(self, X: np.ndarray):
        X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(self.device)
        self._model.eval() 
        with torch.no_grad():
            _, outputs = self._model(X_tensor)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        # Pao a numpy
        pred_idx = predicted.cpu().numpy()[0]
        probs_array = probs.cpu().numpy()[0]

        # Convertir np a formato de salida correcto (con las etiquetas de clase originales)
        class_labels = self.label_encoder.classes_
        prob_dict = {label: float(prob)for label, prob in zip(class_labels, probs_array)}
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        return pred_label, prob_dict
    
    def _build_unprocessed_loader(self, X, Y, batch_size=32, shuffle=False):
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).long()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # ------------------------------------------------------------------
    # Serialización / reconstrucción
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda los pesos actuales del modelo en *path* (fichero .pth).

        Para reconstruir el objeto en otro proceso::

            params = interface.get_constructor_params(path)
            new_interface = MiRepNetInterface(**params)
        """
        torch.save(self._model.state_dict(), path)

    def get_constructor_params(self, weight_path: str) -> dict:
        """Devuelve un dict con todos los argumentos necesarios para reconstruir
        este objeto con ``MiRepNetInterface(**params)``.

        Args:
            weight_path: Ruta al fichero ``.pth`` donde se han guardado los pesos
                         (el devuelto por :meth:`save`).
        """
        return {
            "weight_path":      weight_path,
            "device":           str(self.device),
            "emb_size":         self._emb_size,
            "depth":            self._depth,
            "training_clases":  list(self.training_clases),
        }

    # Getters individuales (útiles si solo se necesita un parámetro)

    def get_emb_size(self) -> int:
        return self._emb_size

    def get_depth(self) -> int:
        return self._depth

    def get_training_clases(self) -> list[str]:
        return list(self.training_clases)

    def get_device(self) -> str:
        return str(self.device)
