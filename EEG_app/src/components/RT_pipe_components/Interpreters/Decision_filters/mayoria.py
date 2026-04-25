from collections import Counter
from .Decision_filter import Decision_filter


class Mayoria(Decision_filter):

    def decider(self, window_probs, window_predictions, window_delays=None, **kwargs):
        counter = Counter(window_predictions)
        pred_final = counter.most_common(1)[0][0]

        info = {
            "name": "Mayoria",
            "n_preds": len(window_predictions),
            "mean_delay": sum(window_delays) / len(window_delays) if window_delays else None,
        }

        return pred_final, info
