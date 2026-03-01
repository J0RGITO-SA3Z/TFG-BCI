"""
PerformanceViewer
=================
Clase para visualizar los resultados de entrenamiento y evaluación
de modelos EEG (MIRepNet y compatibles).

Uso típico
----------
    final_acc, history = model.finetuning(X_train, y_train, epochs=20,
                                           valData=X_val, valLabels=y_val)
    viewer = PerformanceViewer(history)
    viewer.plot_fine_tune()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Paleta compartida ─────────────────────────────────────────────────────────
_C_TRAIN = "#4C72B0"   # azul  — train
_C_VAL   = "#DD8452"   # naranja — validación
_C_LR    = "#55A868"   # verde  — learning rate
_C_BEST  = "gray"      # línea vertical de mejor epoch


class PerformanceViewer:
    """
    Visualizador de métricas de entrenamiento y evaluación.

    Args:
        history : list[dict] devuelta por ``MiRepNetInterface.finetuning``.
                  Cada elemento tiene las claves:
                  ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``, ``lr``.
    """

    def __init__(self, history: list[dict]):
        if not history:
            raise ValueError("El history está vacío.")
        self.history = history
        self._epochs = len(history)
        self._x     = np.arange(1, self._epochs + 1)

        # Extraer series
        self.train_loss = np.array([e["train_loss"] for e in history])
        self.val_loss   = np.array([e["val_loss"]   for e in history])
        self.train_acc  = np.array([e["train_acc"]  for e in history])
        self.val_acc    = np.array([e["val_acc"]    for e in history])
        self.lr         = np.array([e["lr"]         for e in history])

        # Epochs notables
        self.best_loss_epoch = int(np.argmin(self.val_loss))    # índice 0-based
        self.best_acc_epoch  = int(np.argmax(self.val_acc))

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _style_ax(self, ax, xlabel="Epoch", ylabel="", title=""):
        """Aplica estilo común a todos los ejes."""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xticks(self._x)

    def _vline_best(self, ax, epoch_idx, label):
        """Dibuja línea vertical en el mejor epoch y la añade a la leyenda."""
        ax.axvline(
            self._x[epoch_idx],
            color=_C_BEST, linestyle="--", linewidth=1,
            label=label
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def plot_fine_tune(self, show: bool = True):
        """
        Figura con tres subplots:
          1. Loss (train vs val) con línea en el mejor val loss.
          2. Accuracy (train vs val) con línea y anotación en el mejor val acc.
          3. Evolución del learning rate.

        Args:
            show: Si True llama a plt.show() al final.

        Returns:
            fig, axes : objetos matplotlib por si quieres guardar o incrustar.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Evolución del Fine-Tuning", fontsize=14, fontweight="bold")

        ax_loss, ax_acc, ax_lr = axes

        # ── 1. Loss ───────────────────────────────────────────────────────────
        ax_loss.plot(self._x, self.train_loss, color=_C_TRAIN, marker="o",
                     markersize=4, label="Train")
        ax_loss.plot(self._x, self.val_loss,   color=_C_VAL,   marker="o",
                     markersize=4, label="Validación")

        self._vline_best(ax_loss, self.best_loss_epoch,
                         f"Mejor val (epoch {self._x[self.best_loss_epoch]})")

        ax_loss.annotate(
            f"{self.val_loss[self.best_loss_epoch]:.4f}",
            xy=(self._x[self.best_loss_epoch], self.val_loss[self.best_loss_epoch]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, color=_C_VAL, fontweight="bold"
        )

        self._style_ax(ax_loss, ylabel="Loss", title="Loss por epoch")
        ax_loss.legend()

        # ── 2. Accuracy ───────────────────────────────────────────────────────
        ax_acc.plot(self._x, self.train_acc, color=_C_TRAIN, marker="o",
                    markersize=4, label="Train")
        ax_acc.plot(self._x, self.val_acc,   color=_C_VAL,   marker="o",
                    markersize=4, label="Validación")

        self._vline_best(ax_acc, self.best_acc_epoch,
                         f"Mejor val (epoch {self._x[self.best_acc_epoch]})")

        ax_acc.annotate(
            f"{self.val_acc[self.best_acc_epoch]:.1f}%",
            xy=(self._x[self.best_acc_epoch], self.val_acc[self.best_acc_epoch]),
            xytext=(8, -15), textcoords="offset points",
            fontsize=9, color=_C_VAL, fontweight="bold"
        )

        ax_acc.set_ylim(0, 105)
        self._style_ax(ax_acc, ylabel="Accuracy (%)", title="Accuracy por epoch")
        ax_acc.legend()

        # ── 3. Learning rate ──────────────────────────────────────────────────
        ax_lr.plot(self._x, self.lr, color=_C_LR, marker="o", markersize=4,
                   label="Learning rate")
        ax_lr.set_yscale("log")   # log-scale habitual para ver el decay
        self._style_ax(ax_lr, ylabel="LR (escala log)", title="Learning rate por epoch")
        ax_lr.legend()

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axes

    def summary(self):
        """Imprime un resumen compacto de los mejores epochs."""
        print("─" * 50)
        print("  RESUMEN FINE-TUNING")
        print("─" * 50)
        print(f"  Epochs totales     : {self._epochs}")
        print(f"  Mejor val loss     : {self.val_loss[self.best_loss_epoch]:.4f}"
              f"  (epoch {self._x[self.best_loss_epoch]})")
        print(f"  Mejor val accuracy : {self.val_acc[self.best_acc_epoch]:.1f}%"
              f"  (epoch {self._x[self.best_acc_epoch]})")
        print(f"  LR final           : {self.lr[-1]:.6f}")
        print("─" * 50)






