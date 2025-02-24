from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    # root_mean_squared_error,
)
import numpy as np
from ...type_hints import TargetType
from .base import BaseMetric


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

class MSE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False
        self.needs_proba = False
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, y_true: TargetType, y_pred: TargetType, **kwargs):
        return root_mean_squared_error(y_true, y_pred) ** 2

    @property
    def score_name(self) -> str:
        return "MSE"


class MAE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False
        self.needs_proba = False
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, y_true: TargetType, y_pred: TargetType, **kwargs):
        return mean_absolute_error(y_true, y_pred)

    @property
    def score_name(self) -> str:
        return "MAE"


class MAPE(BaseMetric):
    def __init__(self):
        self.greater_is_better = False
        self.needs_proba = False
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, y_true: TargetType, y_pred: TargetType, **kwargs):
        return mean_absolute_percentage_error(y_true, y_pred)

    @property
    def score_name(self) -> str:
        return "MAPE"
