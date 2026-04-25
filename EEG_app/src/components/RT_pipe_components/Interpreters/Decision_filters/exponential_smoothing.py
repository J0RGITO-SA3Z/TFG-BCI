from .Decision_filter import Decision_filter


class ExponentialSmoothing(Decision_filter):

    def __init__(self, alpha=0.85, threshold=0.75):
        """
        alpha: cuánto peso das al pasado (0.85 = muy estable)
        threshold: confianza mínima para decidir
        """
        self.alpha = alpha
        self.threshold = threshold

        # estado interno (memoria temporal)
        self.p_prev = None  # [p_left, p_right]

    def decider(self, window_probs, window_predictions=None, **kwargs):
        if not window_probs:
            return None, {}

        # usamos la última probabilidad de la ventana
        p_left, p_right = window_probs[-1]
        p_t = [float(p_left), float(p_right)]

        # inicialización
        if self.p_prev is None:
            self.p_prev = p_t

        # exponential smoothing
        p_smooth = [
            self.alpha * self.p_prev[0] + (1 - self.alpha) * p_t[0],
            self.alpha * self.p_prev[1] + (1 - self.alpha) * p_t[1],
        ]

        # actualizar estado
        self.p_prev = p_smooth

        # decisión con threshold
        if p_smooth[1] > self.threshold:
            pred_final = "right_hand"
        elif p_smooth[0] > self.threshold:
            pred_final = "left_hand"
        else:
            pred_final = "rest"  # no suficientemente seguro

        info = {
            "name": "ExponentialSmoothing",
            "p_left_smooth": p_smooth[0],
            "p_right_smooth": p_smooth[1],
            "alpha": self.alpha,
            "threshold": self.threshold,
        }

        return pred_final, info