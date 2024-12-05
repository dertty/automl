from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.automl.loggers import get_logger
from src.automl.model.base_model import BaseModel
from src.automl.model.utils import convert_to_numpy, tune_optuna

log = get_logger(__name__)

# class print_with_args:
#     def __init__(self):
#         pass
#     def info(self, msg, **kwargs):
#         print(msg)

# log = print_with_args()


class OptunaBlender(BaseModel):
    def __init__(
        self,
        n=None,
        weights=None,
        n_jobs=1,
        random_state=42,
        drop_thresh=1e-2,
        time_series=False,
    ):
        self.name = "Blender"

        self.n = n
        self.weights = weights

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.drop_thresh = drop_thresh
        self.non_zero_idx = None

        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )

    def fit(self, X, y, categorical_features=None):
        return self._predict(X)

    @staticmethod
    def get_trial_params(trial, n, non_zero_idx):

        weights = np.zeros(n, dtype=np.float32)
        sum_prev_weights = 0

        for idx in non_zero_idx:
            weights[idx] = trial.suggest_float(f"e{idx}", 0.0, 1.0)

        # normalize weights
        weights = weights / np.sum(weights)

        # adjust weights to sum to 1
        weights[non_zero_idx[-1]] = 1 - np.sum(weights[non_zero_idx[:-1]])

        return weights

    def objective(self, trial, X, y, scorer):
        weights = self.get_trial_params(trial, self.n, self.non_zero_idx)

        cv = self.kf.split(X, y)
        cv_metrics = []
        for _, test_idx in cv:
            y_pred = X[test_idx] @ weights

            # if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            #     # binary case
            #     y_pred = y_pred[:, 1]

            cv_metrics.append(scorer.score(y[test_idx], y_pred))

        return np.mean(cv_metrics)

    @staticmethod
    def _compute_weighted_pred(x, weights):
        return np.sum(x * weights.reshape(-1, 1, 1), axis=0)

    def tune(self, X, y, scorer, timeout=60, categorical_features=[]):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        assert X.ndim == 3

        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        timeout = min(timeout, 1)
        self.n = X.shape[-1]
        self.non_zero_idx = np.arange(self.n, dtype=int)
        self.weights = np.array([1 / self.n for _ in range(self.n)])

        first_iter_flag = True
        last_iter_non_zero_idx = deepcopy(self.non_zero_idx)

        while first_iter_flag or (
            np.all(last_iter_non_zero_idx != self.non_zero_idx)
            and len(self.non_zero_idx) != 1
        ):
            # continue iteration if some weights were zerod out on the previous iteration
            if not first_iter_flag:
                log.info(
                    f"Dropped {self.n - len(self.non_zero_idx)} features with weight < {self.drop_thresh}. Continue optimization.",
                    msg_type="optuna",
                )

            study = tune_optuna(
                self.name,
                self.objective,
                X=X,
                y=y,
                scorer=scorer,
                timeout=timeout,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                early_stopping_rounds=None,
                verbose=False,
            )

            for key, val in {
                **study.best_params,
                **study.best_trial.user_attrs,
            }.items():
                idx = int(key[1:])
                self.weights[idx] = val

            # normalize weights
            self.weights = self.weights / np.sum(self.weights)
            # self.weights = np.round(self.weights, 4)

            last_iter_non_zero_idx = deepcopy(self.non_zero_idx)
            self.non_zero_idx = np.where(self.weights >= self.drop_thresh)[0]
            zero_idx = np.setdiff1d(np.arange(self.n), self.non_zero_idx)

            # adjust weights to sum to 1
            self.weights[self.non_zero_idx[-1]] = 1 - np.sum(
                self.weights[self.non_zero_idx[:-1]]
            )

            self.weights[zero_idx] = 0.0

            if len(self.non_zero_idx) == 1:
                # set weight 1 for the only feature
                self.weights[self.non_zero_idx] = 1

            first_iter_flag = False

            log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
            log.info(
                f"Best score {study.best_trial.value} with weights {self.weights.tolist()}",
                msg_type="optuna",
            )

        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        X_test = convert_to_numpy(X_test)
        y = X_test @ self.weights
        y = y.reshape(-1, 1)
        return np.hstack((1 - y, y))

    @property
    def not_tuned_params(self):
        return {"n": self.n, "n_jobs": self.n_jobs, "random_state": self.random_state}

    @property
    def inner_params(self):
        return {"weights": self.weights.tolist(), **self.not_tuned_params}

    @property
    def meta_params(self):
        return {}

    @property
    def params(self):
        return {
            **self.inner_params,
            **self.meta_params,
        }


class CoordDescBlender(BaseModel):
    def __init__(
        self,
        n=None,
        weights=None,
        n_iters=10,
        n_inner_iters=3,
        random_state=42,
        drop_thresh=1e-1,
        time_series=False,
    ):
        """Blender inspired by LightAutoML.
        https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/automl/blend.py
        """
        self.name = "Blender"

        self.n = n
        self.weights = weights

        self.n_iters = n_iters
        self.n_inner_iters = n_inner_iters
        self.random_state = random_state

        self.drop_thresh = drop_thresh

    def fit(self, X, y, categorical_features=None):
        return self._predict(X)

    @staticmethod
    def get_trial_params(trial, idx):
        return trial.suggest_float(f"w{idx}", 0.0, 1.0)

    @staticmethod
    def adjust_weights(weights, idx, drop_thresh):
        residual = max(np.sum(weights) - 1, 0.0)

        adjust_idx = np.delete(np.arange(weights.shape[0]), idx).tolist()
        weights[adjust_idx] = (
            weights[adjust_idx] / np.sum(weights[adjust_idx]) * (1 - weights[idx])
        )

        for i in np.argsort(weights):
            if weights[i] < drop_thresh:
                weights[i] = 0
                weights = weights / np.sum(weights)

        return weights

    def _objective(self, weight, weights, X, y, scorer, idx):

        weights = deepcopy(weights)
        weights[idx] = weight
        weights = self.adjust_weights(weights, idx, self.drop_thresh)
        y_pred = self._compute_weighted_pred(X, weights)
        
        # not_nan_idx = ~np.isnan(y_pred).any(axis=1)
        # y_pred = y_pred[not_nan_idx]

        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            # binary case
            y_pred = y_pred[:, 1]
            
        metric = scorer.score(y, y_pred)
        return -1 * metric if scorer.greater_is_better else metric

    @staticmethod
    def _compute_weighted_pred(x, weights):
        return np.sum(x * weights.reshape(-1, 1, 1), axis=0)

    def tune(self, X, y, scorer, timeout=60, categorical_features=[]):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        assert X.ndim == 3

        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        timeout = min(timeout, 1)
        self.n = X.shape[0]
        self.non_zero_idx = np.arange(self.n, dtype=int)
        self.weights = np.array([1 / self.n for _ in range(self.n)])

        best_score = None
        best_weights = deepcopy(self.weights)

        for i in range(self.n_iters):
            flg_no_upd = True
            for idx in range(self.n):

                weights = deepcopy(self.weights)

                if weights[idx] == 1:
                    continue

                opt_res = minimize_scalar(
                    self._objective,
                    # self._get_scorer(splitted_preds, weights_idx, candidate),
                    method="Bounded",
                    bounds=(0, 1),
                    args=(weights, X, y, scorer, idx),
                    options={"disp": False, "maxiter": self.n_inner_iters},
                )

                weights[idx] = opt_res.x
                weights = self.adjust_weights(weights, idx, self.drop_thresh)

                score = -1 * opt_res.fun if scorer.greater_is_better else opt_res.fun

                if i == 0 and idx == 0:
                    best_score = score
                    best_weights = weights

                elif scorer.is_better(score, best_score):
                    best_score = score
                    best_weights = deepcopy(weights)
                    flg_no_upd = False

                self.weights = weights

            # log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
            log.info(
                f"Best score {best_score} with weights {best_weights.tolist()}",
                msg_type="optuna",
            )

            if flg_no_upd:
                break

        self.weights = best_weights
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        X_test = convert_to_numpy(X_test)
        if X_test.shape[0] != self.n:
            non_zero_idx = np.where(self.weights > 0)[0]

            if X_test.shape[0] == len(non_zero_idx):
                # only the features with non-zero weight are given
                y = self._compute_weighted_pred(X_test, self.weights[non_zero_idx])
            else:
                raise ValueError(
                    f"X contains {X_test.shape[0]} features but should contain {self.n} features."
                )

        else:
            y = self._compute_weighted_pred(X_test, self.weights)
        return y

    @property
    def not_tuned_params(self):
        return {
            "n": self.n,
            "random_state": self.random_state,
            "n_iters": self.n_iters,
            "n_inner_iters": self.n_inner_iters,
        }

    @property
    def inner_params(self):
        return {"weights": self.weights.tolist(), **self.not_tuned_params}

    @property
    def meta_params(self):
        return {}

    @property
    def params(self):
        return {
            **self.inner_params,
            **self.meta_params,
        }
