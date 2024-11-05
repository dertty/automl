import numpy as np
from sklearn.metrics import get_scorer_names


def get_custom_scorer(score_func, greater_is_better):
    class CustomMetric:
        def is_max_optimal(self):
            return greater_is_better

        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            y_pred = np.array(approxes[0]).astype(float)
            y_true = np.array(target).astype(int)

            output_weight = 1 # weight is not used
            
            score = score_func(y_true, y_pred)

            return score, output_weight

        def get_final_error(self, error, weight):
            return error

    return CustomMetric()


def get_eval_metric(scorer):
    score_name = scorer.get_score_name()
    if score_name.lower() in ['roc_auc', 'auc', 'rocauc']:
        return 'AUC'
    elif score_name.lower() in ['accuracy']:
        return 'Accuracy'
    elif score_name.lower() in get_scorer_names():
        return None
    else:
        return get_custom_scorer(scorer.score, scorer.greater_is_better) 



