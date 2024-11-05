import numpy as np
from typing import Optional
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score

from automl.type_hints import TargetType
from .base import BaseMetric


is_one_dimensional = lambda arr: arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
is_binary_classification = lambda arr: arr.ndim == 2 and arr.shape[1] == 2


class Accuracy(BaseMetric):
    def __init__(self, thr=0.5):
        self.greater_is_better = True
        self.needs_proba = True
        self.is_has_thr = True
        self.thr = thr
        self.model_type = None

    def __call__(self, y_true: TargetType, y_pred: TargetType) -> Optional[float]:
        if np.isnan(y_pred).any():
            return None
        
        if is_one_dimensional(y_pred):
            y_pred = y_pred.reshape(-1)
            if np.max(y_pred <= 1) and np.min(y_pred >= 0):
                y_pred = (y_pred > self.thr).astype(int).reshape(-1)
            else:
                # array of labels
                pass
        else:
            # `y_pred` contains probabilities
            # ex.  [[0.1, 0.9],
            #      [0.2, 0.8],
            #      [0.9, 0.1]]
            # convert to labels by applying argmax
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)

    def get_score_name(self) -> str:
        return 'accuracy'
    
    def _get_scorer(self):
        return make_scorer(self, response_method='predict_proba', greater_is_better=self.greater_is_better)


class RocAuc(BaseMetric):
    def __init__(self, multi_class="ovo"):
        self.multi_class = multi_class
        self.greater_is_better = True
        self.needs_proba = True
        self.is_has_thr = False
        self.model_type = None

    def __call__(self, y_true, y_pred) -> Optional[float]:
        if np.isnan(y_pred).any():
            return None
        
        if is_one_dimensional(y_pred):
            self.multi_class = 'raise'
            y_pred = y_pred.reshape(-1)
            if not (np.max(y_pred) <= 1 and np.min(y_pred) >= 0):
                # array of labels
                raise ValueError(
                    "Predictions should contain probabilities for metric RocAuc."
                )
        elif is_binary_classification(y_pred):
            y_pred = y_pred[:, 1]
        else:
            # `y_pred` contains multiclass probabilities
            # ex.  [[0.1, 0.7, 0.2],
            #      [0.05, 0.8, 0.15],
            #      [0.3, 0.1, 0.6]]
            pass

        return roc_auc_score(y_true, y_pred, multi_class=self.multi_class)

    def get_score_name(self):
        return 'roc_auc'
    
    def _get_scorer(self):
        return make_scorer(self, response_method='predict_proba', greater_is_better=self.greater_is_better)