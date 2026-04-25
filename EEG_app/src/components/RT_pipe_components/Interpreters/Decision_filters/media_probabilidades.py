from .Decision_filter import Decision_filter


class MediaProbabilidades(Decision_filter):

    def decider(self, window_probs, window_predictions=None, **kwargs):
        if not window_probs:
            return None, {}

        sum_left = sum(float(p_left) for p_left, _ in window_probs)
        sum_right = sum(float(p_right) for _, p_right in window_probs)

        n = len(window_probs)
        mean_left = sum_left / n
        mean_right = sum_right / n

        pred_final = "right_hand" if mean_right > mean_left else "left_hand"

        info = {
            "mean_left": mean_left,
            "mean_right": mean_right,
            "n_preds": n,
        }

        return pred_final, info
