import numpy as np
import sklearn.metrics as metrics
import automl.metrics.custom_metrics as custom_metrics
from .base import BaseMetric, BaseScorer


class RocAuc(BaseMetric):

    def __init__(self, multi_class="ovo", model_type=None):
        self.needs_proba = True
        self.greater_is_better = True
        self.is_has_thr = False
        self.model_type = model_type
        self.multi_class = multi_class

    def _get_model_score_name(self):
        if self.model_type in ('lightgbm', 'tabnet',):
            model_score_name = 'auc'
        elif self.model_type in ('pytorch_tabular',):
            model_score_name = 'auroc'
        elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
            model_score_name = 'roc_auc'
        elif self.model_type in ('catboost',):
            model_score_name = 'AUC:hints=skip_train~false'
        else:
            model_score_name = None
        return model_score_name

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

        return metrics.roc_auc_score(y_true, y_pred, multi_class=self.multi_class)
    
    def get_scorer(self):
        return BaseScorer(self, "predict_proba")

    @staticmethod
    def _get_score_func(return_thr=False):
        return metrics.roc_auc_score


# class binary_logloss(Metric):

#     def __init__(self, model_type):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name = 'binary_logloss'
#         elif self.model_type == 'tabnet':
#             model_score_name = 'logloss'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name = 'neg_logloss'
#         elif self.model_type == 'catboost':
#             model_score_name = 'Logloss'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'binary_logloss'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return metrics.log_loss


# class prauc(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name =  None
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  None
#         elif self.model_type == 'catboost':
#             model_score_name =  'PRAUC'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'binary_logloss'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return custom_metrics.prauc


# class ap_at_k(Metric):

#     def __init__(self, model_type, k=1.0):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type
#         self.k = k

#     def get_score_name(self):
#         return f'ap_at_{self.k}'

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name =  'average_precision'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  None
#         elif self.model_type == 'catboost':
#             model_score_name =  None
#         else:
#             model_score_name = None
#         return model_score_name

#     def _get_score_func(self, return_thr=False):
#         return lambda y_true, y_pred: custom_metrics.ap_at_k(y_true, y_pred, k=self.k)


# class map_at_k(Metric):

#     def __init__(self, model_type, k=1.0):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type
#         self.k = k

#     def get_score_name(self):
#         return f'map_at_{self.k}'

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name =  'map'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  None
#         elif self.model_type == 'catboost':
#             model_score_name =  None
#         else:
#             model_score_name = None
#         return model_score_name

#     def _get_score_func(self, return_thr=False):
#         return lambda y_true, y_pred: custom_metrics.map_at_k(y_true, y_pred, k=self.k)


# class recall_at_k(Metric):

#     def __init__(self, model_type, k=1.0):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type
#         self.k = k

#     def get_score_name(self):
#         return f'recall_at_{self.k}'

#     def _get_model_score_name(self):
#         return None

#     def _get_score_func(self, return_thr=False):
#         return lambda y_true, y_pred: custom_metrics.recall_at_k(y_true, y_pred, k=self.k)


# class precision_at_k(Metric):

#     def __init__(self, model_type, k=1.0):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type
#         self.k = k

#     def get_score_name(self):
#         return f'precision_at_{self.k}'

#     def _get_model_score_name(self):
#         return None

#     def _get_score_func(self, return_thr=False):
#         return lambda y_true, y_pred: custom_metrics.precision_at_k(y_true, y_pred, k=self.k)


# class recall(Metric):

#     def __init__(self, model_type, thr=0.5):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = True
#         self.model_type = model_type
#         self.thr = thr

#     def get_score_name(self):
#         return 'recall'

#     def _get_model_score_name(self):
#         return None

#     def _get_score_func(self, return_thr=False):
#         return lambda y_true, y_pred: custom_metrics.recall(y_true, y_pred, thr=self.thr, return_thr=return_thr)


# class precision(Metric):

#     def __init__(self, model_type, thr=0.5):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = True
#         self.model_type = model_type
#         self.thr = thr

#     def get_score_name(self):
#         return 'precision'

#     def _get_model_score_name(self):
#         return None

#     def _get_score_func(self, return_thr=True):
#         return lambda y_true, y_pred: custom_metrics.precision(y_true, y_pred, thr=self.thr, return_thr=return_thr)


# class f(Metric):

#     def __init__(self, model_type, beta=1.0, thr=0.5):
#         self.needs_proba = True
#         self.greater_is_better = True
#         self.is_has_thr = True
#         self.model_type = model_type
#         self.beta = beta
#         self.thr = thr

#     def get_score_name(self):
#         return f'f_{self.beta}'

#     def _get_model_score_name(self):
#         return None

#     def _get_score_func(self, return_thr=True):
#         return lambda y_true, y_pred: custom_metrics.f_beta(y_true, y_pred, beta=self.beta, thr=self.thr, return_thr=return_thr)


# class mae(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = False
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type in ['lightgbm', 'tabnet',]:
#             model_score_name =  'mae'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  'neg_mean_absolute_error'
#         elif self.model_type == 'catboost':
#             model_score_name =  'MAE'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'mae'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return metrics.mean_absolute_error


# class mse(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = False
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type in ['lightgbm','tabnet',]:
#             model_score_name =  'mse'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  'neg_mean_squared_error'
#         elif self.model_type == 'catboost':
#             model_score_name =  None
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'mse'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred, squared=True)


# class rmse(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = False
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type in ['lightgbm','tabnet',]:
#             model_score_name =  'rmse'
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  'neg_root_mean_squared_error'
#         elif self.model_type == 'catboost':
#             model_score_name =  'RMSE'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'rmse'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred, squared=False)


# class mape(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = False
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name =  'mape'
#         elif self.model_type == 'tabnet':
#             model_score_name =  None
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  'neg_mean_absolute_percentage_error'
#         elif self.model_type == 'catboost':
#             model_score_name =  'MAPE'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'mape'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return metrics.mean_absolute_percentage_error


# class smape(Metric):
    
#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = False
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         if self.model_type == 'lightgbm':
#             model_score_name =  None
#         elif self.model_type in ('extratree', 'randomforest') or 'sklearn' in self.model_type:
#             model_score_name =  None
#         elif self.model_type == 'catboost':
#             model_score_name =  'SMAPE'
#         else:
#             model_score_name = None
#         return model_score_name

#     def get_score_name(self):
#         return 'smape'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return custom_metrics.symmetric_mean_absolute_percentage_error


# class regression_roc_auc(Metric):

#     def __init__(self, model_type):
#         self.needs_proba = False
#         self.greater_is_better = True
#         self.is_has_thr = False
#         self.model_type = model_type

#     def _get_model_score_name(self):
#         return None

#     def get_score_name(self):
#         return 'reg_roc_auc'

#     @staticmethod
#     def _get_score_func(return_thr=False):
#         return custom_metrics.regression_roc_auc_score