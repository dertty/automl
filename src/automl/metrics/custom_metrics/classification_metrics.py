import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score

from automl.type_hints import TargetType
from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, thr=0.5):
        self.greater_is_better = True
        self.needs_proba = True
        self.is_has_thr = True
        self.thr = thr
        self.model_type = None

    def __call__(self, y_true: TargetType, y_pred: TargetType):
        if y_pred.ndim == 1:
            # y_pred is an array of labels (ex. [1, 0, 2, 1, 2, 0])
            # or an array of binary probabilities (ex. [0.8, 0.1, 0.4, 0.9])

            if np.max(y_pred < 1):
                # binary probabilities
                # convert to labels and reshape
                y_pred = (y_pred > self.thr).reshape(y_pred.shape[0])
            else:
                # array of labels
                pass

        elif y_pred.ndim > 1:
            # y_pred contains probabilities
            if y_pred.shape[1] == 1:
                # `y_pred` contains probabilities of 1 class
                # ex. [[0.1],
                #      [0.2],
                #      [0.9]]
                # compare `y_pred` with 0.5 and flatten an array
                y_pred = (y_pred > self.thr).astype(int).reshape(y_pred.shape[0])

            elif y_pred.shape[1] > 1:
                # `y_pred` contains probabilities
                # ex.  [[0.1, 0.9],
                #      [0.2, 0.8],
                #      [0.9, 0.1]]
                # convert to labels by applying argmax
                y_pred = np.argmax(y_pred, axis=1)

        if np.isnan(y_pred).any():
            return None

        return accuracy_score(y_true, y_pred)

    def get_score_name(self):
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

    def __call__(self, y_true, y_pred):

        if y_pred.ndim == 1:
            # y_pred is an array of labels (ex. [1, 0, 2, 1, 2, 0])
            # or an array of binary probabilities (ex. [0.8, 0.1, 0.4, 0.9])

            if np.max(y_pred < 1):
                # binary probabilities
                # reshape to column format
                y_pred = y_pred.reshape(y_pred.shape[0])
            else:
                # array of labels
                raise ValueError(
                    "Predictions should contain probabilities for metric RocAuc."
                )

        elif y_pred.ndim > 1:
            # y_pred contains probabilities
            if y_pred.shape[1] == 1:
                # `y_pred` contains probabilities of 1 class
                # ex. [[0.1],
                #      [0.2],
                #      [0.9]]
                pass

            elif y_pred.shape[1] == 2:
                # `y_pred` contains probabilities of 0 and 1 class
                # ex.  [[0.1, 0.9],
                #      [0.2, 0.8],
                #      [0.9, 0.1]]
                # take only the probabilities of a 1-st class
                y_pred = y_pred[:, [1]]

            elif y_pred.shape[1] > 2:
                # `y_pred` contains multiclass probabilities
                # ex.  [[0.1, 0.7, 0.2],
                #      [0.05, 0.8, 0.15],
                #      [0.3, 0.1, 0.6]]
                pass

        if np.isnan(y_pred).any():
            return None

        return roc_auc_score(y_true, y_pred, multi_class=self.multi_class)

    def get_score_name(self):
        return 'roc_auc'
    
    def _get_scorer(self):
        return make_scorer(self, response_method='predict_proba', greater_is_better=self.greater_is_better)