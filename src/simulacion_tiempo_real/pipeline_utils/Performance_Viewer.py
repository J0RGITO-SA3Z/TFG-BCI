"""
PerformanceViewer
=================
Clase de utilidades para visualizar métricas de entrenamiento y evaluación
de modelos EEG (MIRepNet y compatibles).

Uso típico
----------
    result = model.finetuning(X_train, y_train, epochs=20,
                              valData=X_val, valLabels=y_val)

    viewer = PerformanceViewer()
    viewer.plot_fine_tune(result)   # result puede ser la tupla (acc, history) o el history directamente
    viewer.summary(result)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from matplotlib.widgets import Slider

# ── Paleta compartida ─────────────────────────────────────────────────────────
_C_TRAIN = "#4C72B0"   # azul    — train
_C_VAL   = "#DD8452"   # naranja — validación
_C_LR    = "#55A868"   # verde   — learning rate
_C_BEST  = "gray"      # línea vertical mejor epoch


class PerformanceViewer:
    """
    Clase de funciones para visualizar métricas de entrenamiento y evaluación.
    No requiere datos en el constructor; se pasan como argumento a cada método.
    """

    def __init__(self):
        pass

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _parse_history(self, result):
        """
        Acepta la tupla devuelta por finetuning() → (final_acc, history)
        o directamente el history (list[dict]).
        Siempre devuelve el history.
        """
        if isinstance(result, tuple):
            history = result[1]
        else:
            history = result

        if not history:
            raise ValueError("El history está vacío.")
        return history

    def _build_series(self, history):
        """Extrae arrays numpy del history."""
        x = np.arange(1, len(history) + 1)
        return (
            x,
            np.array([e["train_loss"] for e in history]),
            np.array([e["val_loss"]   for e in history]),
            np.array([e["train_acc"]  for e in history]),
            np.array([e["val_acc"]    for e in history]),
            np.array([e["lr"]         for e in history]),
        )

    def _style_ax(self, ax, x, ylabel="", title=""):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xticks(x)

    def _vline_best(self, ax, x, epoch_idx, label):
        ax.axvline(x[epoch_idx], color=_C_BEST, linestyle="--", linewidth=1, label=label)

    # ── metodos públicos ───────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # FINE-TUNING
    # ─────────────────────────────────────────────────────────────────────────

    def plot_fine_tune(self, result, show: bool = True):
        """
        Figura con tres subplots: Loss, Accuracy y Learning Rate.

        Args:
            result : tupla (final_acc, history) devuelta por finetuning(),
                     o directamente el history (list[dict]).
            show   : Si True llama a plt.show() al final.

        Returns:
            fig, axes
        """
        history = self._parse_history(result)
        x, train_loss, val_loss, train_acc, val_acc, lr = self._build_series(history)

        best_loss_epoch = int(np.argmin(val_loss))
        best_acc_epoch  = int(np.argmax(val_acc))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Evolución del Fine-Tuning", fontsize=14, fontweight="bold")
        ax_loss, ax_acc, ax_lr = axes

        # ── 1. Loss ───────────────────────────────────────────────────────────
        ax_loss.plot(x, train_loss, color=_C_TRAIN, marker="o", markersize=4, label="Train")
        ax_loss.plot(x, val_loss,   color=_C_VAL,   marker="o", markersize=4, label="Validación")
        self._vline_best(ax_loss, x, best_loss_epoch,
                         f"Mejor val (epoch {x[best_loss_epoch]})")
        ax_loss.annotate(
            f"{val_loss[best_loss_epoch]:.4f}",
            xy=(x[best_loss_epoch], val_loss[best_loss_epoch]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, color=_C_VAL, fontweight="bold"
        )
        self._style_ax(ax_loss, x, ylabel="Loss", title="Loss por epoch")
        ax_loss.legend()

        # ── 2. Accuracy ───────────────────────────────────────────────────────
        ax_acc.plot(x, train_acc, color=_C_TRAIN, marker="o", markersize=4, label="Train")
        ax_acc.plot(x, val_acc, color=_C_VAL,   marker="o", markersize=4, label="Validación")
        self._vline_best(ax_acc, x, best_acc_epoch,
                         f"Mejor val (epoch {x[best_acc_epoch]})")
        ax_acc.annotate(
            f"{val_acc[best_acc_epoch]:.1f}%",
            xy=(x[best_acc_epoch], val_acc[best_acc_epoch]),
            xytext=(8, -15), textcoords="offset points",
            fontsize=9, color=_C_VAL, fontweight="bold"
        )
        ax_acc.set_ylim(0, 105)
        self._style_ax(ax_acc, x, ylabel="Accuracy (%)", title="Accuracy por epoch")
        ax_acc.legend()

        # ── 3. Learning rate ──────────────────────────────────────────────────
        ax_lr.plot(x, lr, color=_C_LR, marker="o", markersize=4, label="Learning rate")
        ax_lr.set_yscale("log")
        self._style_ax(ax_lr, x, ylabel="LR (escala log)", title="Learning rate por epoch")
        ax_lr.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axes

    def summary(self, result):
        """Imprime un resumen compacto de los mejores epochs."""
        history = self._parse_history(result)
        x, _, val_loss, _, val_acc, lr = self._build_series(history)

        best_loss_epoch = int(np.argmin(val_loss))
        best_acc_epoch  = int(np.argmax(val_acc))

        print("─" * 50)
        print("  RESUMEN FINE-TUNING")
        print("─" * 50)
        print(f"  Epochs totales     : {len(history)}")
        print(f"  Mejor val loss     : {val_loss[best_loss_epoch]:.4f}"
              f"  (epoch {x[best_loss_epoch]})")
        print(f"  Mejor val accuracy : {val_acc[best_acc_epoch] :.1f}%"
              f"  (epoch {x[best_acc_epoch]})")
        print(f"  LR final           : {lr[-1]:.6f}")
        print("─" * 50)

    def plot_downstream(self, y_pred, y_probs, y_true):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14,8))

        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1, 2],
            height_ratios=[1, 1]
        )

        # izquierda ocupa ambas filas
        ax_cm = fig.add_subplot(gs[:, 0])

        # derecha
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_conf = fig.add_subplot(gs[1, 1])

        # dibujar (adaptado a nuevo formato)
        self.plot_matriz_confusion(
            y_pred, y_true,
            ax=ax_cm
        )

        self.plot_metricas_clase(
            y_pred, y_true,
            ax=ax_metrics
        )

        self.plot_confianza(
            y_pred, y_probs, y_true,
            ax=ax_conf
        )

        # mantener cuadrada la matriz de confusión
        ax_cm.set_aspect('equal', adjustable='box', anchor='W')

        plt.tight_layout()
        plt.show()
    
    def plot_matriz_confusion(self, y_pred, y_true, ax=None):
        """
        Matriz de confusión usando etiquetas (strings).
        """

        # Obtener clases automáticamente (orden consistente)
        labels = sorted(list(set(y_true) | set(y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )

        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title("Confusion Matrix")

        return cm
    
    def plot_metricas_clase(self, y_pred, y_true, ax=None):
        """
        Métricas por clase usando etiquetas (strings).
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Obtener clases automáticamente (orden consistente)
        class_names = sorted(list(set(y_true) | set(y_pred)))

        # Precision y recall por clase
        precision = precision_score(
            y_true, y_pred, labels=class_names, average=None, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, labels=class_names, average=None, zero_division=0
        )

        # Accuracy por clase
        acc_por_clase = []
        for c in class_names:
            idx = y_true == c
            if np.sum(idx) == 0:
                acc_por_clase.append(0.0)
            else:
                acc = np.mean(y_pred[idx] == y_true[idx])
                acc_por_clase.append(acc)

        acc_por_clase = np.array(acc_por_clase)

        # Accuracy global
        acc_total = np.mean(y_pred == y_true)

        x = np.arange(len(class_names))
        width = 0.25

        # Crear figura si no se pasa ax
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
            created_fig = True

        # Barras
        ax.bar(x - width, acc_por_clase, width, label="Accuracy")
        ax.bar(x, precision, width, label="Precision")
        ax.bar(x + width, recall, width, label="Recall")

        # Línea accuracy total
        ax.axhline(
            acc_total,
            linestyle="--",
            linewidth=2,
            label=f"Accuracy total = {acc_total:.2f}"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names)

        ax.set_ylabel("Score")
        ax.set_ylim(0,1)

        ax.set_title("Metricas por clase")
        ax.legend()

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_confianza(self,y_pred, y_probs, y_true, ax=None):
        """
        Distribución de confianza del modelo por clase (correctos vs errores).
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # confianza = probabilidad de la clase predicha
        confianza = np.array([
            prob_dict[pred]
            for pred, prob_dict in zip(y_pred, y_probs)
        ])

        # clases presentes
        class_names = sorted(list(set(y_true) | set(y_pred)))

        data = []
        labels = []

        for c in class_names:

            idx = y_pred == c

            correct = confianza[idx & (y_true == c)]
            errors  = confianza[idx & (y_true != c)]

            data.append(correct)
            data.append(errors)

            labels.append(f"{c}\nCorrect")
            labels.append(f"{c}\nError")

        # Crear figura si no se pasa ax
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,5))
            created_fig = True

        box = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )

        # colores: verde = correcto, rojo = error
        colors = ["green","red"] * len(class_names)

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)

        ax.set_ylabel("Model confidence")
        ax.set_ylim(0,1)

        ax.set_title("Confidence distribution per class")

        if created_fig:
            plt.tight_layout()
            plt.show()