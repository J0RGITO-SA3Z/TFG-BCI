from .Decision_filter import Decision_filter


class ScorePonderado(Decision_filter):

    def decider(self, window_probs, window_predictions, **kwargs):
        score_total = 0
        scores = []

        for (p_left, p_right), pred in zip(window_probs, window_predictions):
            score = abs(float(p_right) - float(p_left))
            if pred == "right_hand":
                score_total += score
            else:
                score_total -= score
            scores.append(score)

        pred_final = "right_hand" if score_total > 0 else "left_hand"

        info = {
            "score_total": score_total,
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "n_preds": len(scores),
        }

        return pred_final, info
