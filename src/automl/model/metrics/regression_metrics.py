from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from ..type_hints import TargetType
from .base import BaseMetric, BaseScorer


class MSE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return mean_squared_error(y_true, y_pred)

    def get_scorer(self):
        return BaseScorer(self, "predict")


class MAE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return mean_absolute_error(y_true, y_pred)

    def get_scorer(self):
        return BaseScorer(self, "predict")


class MAPE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        return mean_absolute_percentage_error(y_true, y_pred)

    def get_scorer(self):
        return BaseScorer(self, "predict")
