import numpy as np
from sklearn.metrics import make_scorer


def get_catboost_custom_scorer(score_func, greater_is_better):
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


def get_scorer(model_type, score_func, score_name, needs_proba, greater_is_better):
    if model_type == 'sklearn':
        return make_scorer(score_func, needs_proba=needs_proba, greater_is_better=greater_is_better)
    elif model_type == 'lightgbm':
        return lambda y_pred, y_true: (score_name, score_func(y_true.get_label(), y_pred), greater_is_better)
    elif model_type == 'catboost':
        return get_catboost_custom_scorer(score_func, greater_is_better)
    else:
        return None


class Metric:
    def __init__(self, model_type):
        self.needs_proba = True
        self.greater_is_better = True
        self.is_has_thr = False
        self.model_type = model_type

    def _get_model_score_name(self):
        return None
    
    def get_score_name(self):
        return ''

    def _get_score_func(self, return_thr=False):
        return lambda y_true, y_pred: 0

    def get_scorer(self):
        return get_scorer(self.model_type, self._get_score_func(return_thr=False), self.get_score_name(), self.needs_proba, self.greater_is_better)
    
    def score(self, y_true, y_pred):
        score_func = self._get_score_func(return_thr=True)
        if self.is_has_thr:
            score, self.thr = score_func(y_true, y_pred)
        else:
            score = score_func(y_true, y_pred)
        return score
    
    def set_thr(self, thr):
        self.thr = thr

    def get_thr(self):
        return self.thr