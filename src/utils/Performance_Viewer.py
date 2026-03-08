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

    
    # ─────────────────────────────────────────────────────────────────────────
    # DOWNSTREAM
    # ─────────────────────────────────────────────────────────────────────────

    def plot_downstream(self, true_labels, probs,
                        class_names=("feet", "left_hand", "right_hand"),
                        show: bool = True):
        """
        Figura con cuatro subplots para evaluar predict_batch():
          1. Predicción trial a trial (verde=acierto, rojo=fallo).
          2. Matriz de confusión.
          3. Accuracy por clase.
          4. Confianza máxima del modelo por trial.

        Args:
            true_labels : list[str] — etiquetas reales en formato modelo
                          ("feet", "left_hand", "right_hand").
            probs       : np.ndarray (B, 3) — salida de predict_batch().
            class_names : secuencia con los nombres de las 3 clases (mismo
                          orden que las columnas de probs).
            show        : Si True llama a plt.show() al final.

        Returns:
            fig, axes
        """
        class_names  = list(class_names)
        label_to_idx = {c: i for i, c in enumerate(class_names)}

        # Derivar predicciones y confianza desde probs
        pred_idx    = probs.argmax(axis=1)                          # (B,)
        pred_labels = [class_names[i] for i in pred_idx]
        max_conf    = probs.max(axis=1)                             # (B,)

        n        = len(true_labels)
        correct  = [t == p for t, p in zip(true_labels, pred_labels)]
        accuracy = sum(correct) / n * 100

        colors_map = {
            "left_hand"  : "#4C72B0",
            "right_hand" : "#DD8452",
            "feet"       : "#55A868",
        }

        bar_colors = [colors_map.get(c, "#999999") for c in class_names]

        fig = plt.figure(figsize=(16, 15))
        fig.suptitle("Resultados — Downstream Evaluation",
                     fontsize=15, fontweight="bold")

        x = np.arange(n)

        # ── Subplot 1: trial a trial (fila 1, ocupa ambas columnas) ──────────
        ax1 = fig.add_subplot(3, 2, (1, 2))

        true_idx_plot = [label_to_idx.get(l, 0) for l in true_labels]

        for i in range(n):
            color = "#2ecc71" if correct[i] else "#e74c3c"
            ax1.scatter(i, true_idx_plot[i], marker="o", s=120, color=color,
                        zorder=3, edgecolors="white", linewidths=0.8)
            if not correct[i]:
                ax1.annotate(
                    pred_labels[i],
                    xy=(i, true_idx_plot[i]),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=7, color="#e74c3c",
                    arrowprops=dict(arrowstyle="-", color="#e74c3c", lw=0.8)
                )

        ax1.set_yticks(range(len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.set_xlabel("Trial")
        ax1.set_title("Predicción por trial  (verde=acierto, rojo=fallo — texto indica predicción errónea)")
        ax1.grid(axis="x", alpha=0.3)
        ax1.set_xticks(x)
        ax1.legend(handles=[
            mpatches.Patch(color="#2ecc71", label="Acierto"),
            mpatches.Patch(color="#e74c3c", label="Fallo"),
        ], loc="upper right")

        # ── Subplot 2: matriz de confusión (fila 3, col 1) ───────────────────
        ax2 = fig.add_subplot(3, 2, 5)

        nc          = len(class_names)
        conf_matrix = np.zeros((nc, nc), dtype=int)
        for t, p in zip(true_labels, pred_labels):
            conf_matrix[label_to_idx.get(t, 0), label_to_idx.get(p, 0)] += 1

        ax2.imshow(conf_matrix, cmap="Blues")
        ax2.set_xticks(range(nc))
        ax2.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
        ax2.set_yticks(range(nc))
        ax2.set_yticklabels(class_names, fontsize=9)
        ax2.set_xlabel("Predicción")
        ax2.set_ylabel("Real")
        ax2.set_title("Matriz de confusión")
        for i in range(nc):
            for j in range(nc):
                ax2.text(j, i, str(conf_matrix[i, j]),
                         ha="center", va="center", fontsize=11, fontweight="bold",
                         color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

        # ── Subplot 3: accuracy por clase (fila 3, col 2) ────────────────────
        ax3 = fig.add_subplot(3, 2, 6)

        acc_per_class = []
        for c in class_names:
            indices = [i for i, t in enumerate(true_labels) if t == c]
            acc_per_class.append(
                sum(correct[i] for i in indices) / len(indices) * 100 if indices else 0.0
            )

        bars = ax3.bar(class_names, acc_per_class, color=bar_colors,
                       edgecolor="white", linewidth=1.2)
        ax3.axhline(accuracy, color="black", linestyle="--", linewidth=1.5,
                    label=f"Total: {accuracy:.1f}%")
        ax3.set_ylim(0, 110)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Accuracy por clase")
        ax3.legend(fontsize=10)
        for bar, val in zip(bars, acc_per_class):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

        plt.tight_layout()

        # ── Subplot 4: confianza por trial (fila 2, ocupa ambas columnas) ────
        ax_conf = fig.add_subplot(3, 2, (3, 4))

        conf_colors = ["#2ecc71" if c else "#e74c3c" for c in correct]
        ax_conf.bar(x, max_conf * 100, color=conf_colors,
                    edgecolor="white", linewidth=0.6)
        ax_conf.axhline(100 / len(class_names), color="gray", linestyle="--",
                        linewidth=1, label=f"Azar ({100/len(class_names):.0f}%)")
        ax_conf.set_xlabel("Trial")
        ax_conf.set_ylabel("Confianza máxima (%)")
        ax_conf.set_title("Confianza del modelo por trial  (verde=acierto, rojo=fallo)")
        ax_conf.set_ylim(0, 105)
        ax_conf.set_xticks(x)
        ax_conf.grid(axis="y", alpha=0.3)
        ax_conf.legend(handles=[
            mpatches.Patch(color="#2ecc71", label="Acierto"),
            mpatches.Patch(color="#e74c3c", label="Fallo"),
            plt.Line2D([0], [0], color="gray", linestyle="--",
                       label=f"Azar ({100/len(class_names):.0f}%)"),
        ], fontsize=9)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, (ax1, ax_conf, ax2, ax3)
        
    def plot_downstream2(self, y_softmax, y_true, class_names=None):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14,8))

        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1, 2],   # la derecha absorbe el ancho extra
            height_ratios=[1, 1]
        )

        # izquierda ocupa ambas filas
        ax_cm = fig.add_subplot(gs[:, 0])

        # derecha
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_conf = fig.add_subplot(gs[1, 1])

        # dibujar
        self.plot_matriz_confusion(
            y_softmax, y_true,
            class_names=class_names,
            ax=ax_cm
        )

        self.plot_metricas_clase(
            y_softmax, y_true,
            class_names=class_names,
            ax=ax_metrics
        )

        self.plot_confianza(
            y_softmax, y_true,
            class_names=class_names,
            ax=ax_conf
        )

        # mantener cuadrada y usar todo el alto
        ax_cm.set_aspect('equal', adjustable='box', anchor='W')

        plt.tight_layout()
        plt.show()
    
    def plot_matriz_confusion(self, y_softmax, y_true, class_names=None, ax=None):

        y_pred = np.argmax(y_softmax, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title("Confusion Matrix")

        return cm
    
    def plot_metricas_clase(self, y_softmax, y_true, class_names=None, ax=None):

        y_true = np.asarray(y_true)
        y_softmax = np.asarray(y_softmax)

        # convertir softmax -> clase predicha
        y_pred = np.argmax(y_softmax, axis=1)

        clases = np.unique(y_true)

        acc_por_clase = []
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)

        for c in clases:
            idx = y_true == c
            acc = np.mean(y_pred[idx] == y_true[idx])
            acc_por_clase.append(acc)

        acc_por_clase = np.array(acc_por_clase)

        # accuracy global
        acc_total = np.mean(y_pred == y_true)

        if class_names is None:
            class_names = [str(c) for c in clases]

        x = np.arange(len(class_names))
        width = 0.25

        # si no hay eje, crear figura propia
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
            created_fig = True

        # barras
        ax.bar(x - width, acc_por_clase, width, label="Accuracy")
        ax.bar(x, precision, width, label="Precision")
        ax.bar(x + width, recall, width, label="Recall")

        # linea accuracy total
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

    def plot_confianza(self, y_softmax, y_true, class_names=None, ax=None):

        y_true = np.asarray(y_true)
        y_softmax = np.asarray(y_softmax)

        y_pred = np.argmax(y_softmax, axis=1)
        confianza = np.max(y_softmax, axis=1)

        clases = np.unique(y_true)

        data = []
        labels = []

        for c in clases:

            idx = y_pred == c

            correct = confianza[idx & (y_true == c)]
            errors  = confianza[idx & (y_true != c)]

            data.append(correct)
            data.append(errors)

            if class_names:
                name = class_names[c]
            else:
                name = str(c)

            labels.append(f"{name}\nCorrect")
            labels.append(f"{name}\nError")

        # crear figura solo si no se pasa ax
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,5))
            created_fig = True

        box = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )

        colors = ["green","red"] * len(clases)

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