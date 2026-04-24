from .Decision_filter import Decision_filter


class IntegradorFuga(Decision_filter):

    def __init__(self, leak_r=0.1, leak_l=0.1, threshold=1.0, reset_on_decision=False):
        super().__init__()
        self.leak_r = leak_r
        self.leak_l = leak_l
        self.threshold = threshold
        self.reset_on_decision = reset_on_decision
        self.integrator = 0.0

    def decider(self, window_probs, window_predictions=None, **kwargs):
        if self.integrator > 0:
            self.integrator = max(0.0, self.integrator - self.leak_r)
        else:
            self.integrator = min(0.0, self.integrator + self.leak_l)

        for p_left, p_right in window_probs:
            self.integrator += float(p_right) - float(p_left)

        if self.integrator > self.threshold:
            pred_final = "right_hand"
            if self.reset_on_decision:
                self.integrator = 0.0
        elif self.integrator < -self.threshold:
            pred_final = "left_hand"
            if self.reset_on_decision:
                self.integrator = 0.0
        else:
            pred_final = None

        info = {
            "integrator": self.integrator,
            "threshold": self.threshold,
        }

        return pred_final, info
