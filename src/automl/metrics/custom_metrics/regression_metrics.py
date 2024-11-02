from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.metrics import make_scorer
from automl.type_hints import TargetType
from .base import BaseMetric


class MSE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return root_mean_squared_error(y_true, y_pred) ** 2

    def get_scorer(self):
        return make_scorer(self, response_method='predict_proba', greater_is_better=self.greater_is_better)


class MAE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False
        self.needs_proba = False
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return mean_absolute_error(y_true, y_pred)

    def _get_scorer(self):
        return make_scorer(self, response_method='predict', greater_is_better=self.greater_is_better)
    

class MAPE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False
        self.needs_proba = False
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return mean_absolute_percentage_error(y_true, y_pred)

    def _get_scorer(self):
        return make_scorer(self, response_method='predict', greater_is_better=self.greater_is_better)
