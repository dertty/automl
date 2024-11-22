from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from ...type_hints import TargetType
from .base import BaseMetric

is_one_dimensional = lambda arr: arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
is_binary_classification = lambda arr: arr.ndim == 2 and arr.shape[1] == 2


class Accuracy(BaseMetric):
    def __init__(self, thr=0.5):
        self.greater_is_better = True
        self.needs_proba = False
        self.is_has_thr = True
        self.thr = thr
        self.model_type = None

    def __call__(
        self, y_true: TargetType, y_pred: TargetType, **kwargs
    ) -> Optional[float]:
        if np.isnan(y_pred).any():
            return None

        if is_one_dimensional(y_pred):
            # y_pred is an array of labels (ex. [1, 0, 2, 1, 2, 0])
            # or an array of binary probabilities (ex. [0.8, 0.1, 0.4, 0.9])
            # or `y_pred` contains probabilities of 1 class
            #  ex. [[0.1],
            #      [0.2],
            #      [0.9]]
            y_pred = y_pred.reshape(-1)

            if np.max(y_pred <= 1) and np.min(y_pred >= 0):
                # binary probabilities
                # convert to labels
                y_pred = (y_pred > self.thr).astype(int)
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

    @property
    def score_name(self) -> str:
        return "accuracy"


class RocAuc(BaseMetric):
    def __init__(self, multi_class="ovo"):
        self.multi_class = multi_class
        self.greater_is_better = True
        self.needs_proba = True
        self.is_has_thr = False
        self.model_type = None

    def __call__(self, y_true, y_pred, **kwargs) -> Optional[float]:
        if np.isnan(y_pred).any():
            return None

        if is_one_dimensional(y_pred):
            # y_pred is an array of labels (ex. [1, 0, 2, 1, 2, 0])
            # or an array of binary probabilities (ex. [0.8, 0.1, 0.4, 0.9])
            # or `y_pred` contains probabilities of 1 class
            #  ex. [[0.1],
            #      [0.2],
            #      [0.9]]
            self.multi_class = "raise"
            y_pred = y_pred.reshape(-1)

            if np.max(y_pred) <= 1 and np.min(y_pred) >= 0:
                # binary probabilities
                pass
            else:
                # array of labels
                raise ValueError(
                    "Predictions should contain probabilities for metric RocAuc."
                )

        elif is_binary_classification(y_pred):
            # `y_pred` contains probabilities of 0 and 1 class
            # ex.  [[0.1, 0.9],
            #      [0.2, 0.8],
            #      [0.9, 0.1]]
            # take only the probabilities of a 1-st class
            y_pred = y_pred[:, 1]
        else:
            # `y_pred` contains multiclass probabilities
            # ex.  [[0.1, 0.7, 0.2],
            #      [0.05, 0.8, 0.15],
            #      [0.3, 0.1, 0.6]]
            pass

        return roc_auc_score(y_true, y_pred, multi_class=self.multi_class)

    @property
    def score_name(self):
        return "roc_auc"
