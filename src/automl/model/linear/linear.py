import warnings
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression as LogRegSklearn
from sklearn.linear_model import LogisticRegressionCV, Ridge, RidgeCV
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)

from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import SuppressWarnings, convert_to_numpy

warnings.filterwarnings("ignore")


log = get_logger(__name__)


class RidgeRegression(BaseModel):
    def __init__(
        self,
        alpha=None,
        tune_alphas: List[float] = [0.1, 0.5, 1, 5, 10, 50],
        random_state: int = 42,
        time_series=False,
        n_jobs=None,
    ):

        self.name = "RidgeRegression"

        self.alpha = alpha
        self.tune_alphas = tune_alphas
        self.random_state = random_state
        self.time_series = time_series
        self.n_jobs = None

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        self.models = []
        cv = self.kf.split(X)
        oof_preds = np.full(y.shape[0], fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            # initialize fold model
            fold_model = Ridge(self.alpha, random_state=self.random_state)

            # fit/predict fold model
            fold_model.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = fold_model.predict(X[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=None,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X)

        ridge_cv = RidgeCV(alphas=self.tune_alphas, cv=cv).fit(X, y)
        self.alpha = float(ridge_cv.alpha_)

        log.info(f"{self.params}", msg_type="best_params")

        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = X_test.to_numpy()

        y_pred = np.zeros(X_test.shape[0])
        for fold_model in self.models:
            y_pred += fold_model.predict(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def params(self):
        return {"alpha": self.alpha, "random_state": self.random_state}


class LogisticRegression(BaseModel):
    def __init__(
        self,
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        n_jobs=6,
        random_state: int = 42,
        time_series=False,
    ):

        self.name = "LogisticRegression"

        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        self.models = []
        cv = self.kf.split(X, y)
        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            # initialize fold model
            fold_model = LogRegSklearn(**self.inner_params)

            # suppress annoying sklearn ConvergenceWarning
            with SuppressWarnings():
                # fit/predict fold model
                fold_model.fit(X[train_idx], y[train_idx])

            oof_preds[test_idx] = fold_model.predict_proba(X[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=None,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X, y)

        # suppress annoying sklearn ConvergenceWarning
        with SuppressWarnings():
            # cross validate logistic regression to find best C
            logistic_cv = LogisticRegressionCV(
                class_weight=self.class_weight,
                cv=cv,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ).fit(X, y)

        # obtain the list of best Cs. One C for each class.
        # read more at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        Cs = logistic_cv.C_
        # log.info(f"Choosing from {Cs}", msg_type="best_params")

        # iterate over all found Cs and find the one maximizing the metric
        best_metric = None
        best_C = None
        for i, C in enumerate(Cs):
            cv = self.kf.split(X, y)

            # initialzie the model
            model = LogRegSklearn(
                C=C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

            # suppress annoying sklearn ConvergenceWarning
            with SuppressWarnings():
                # cross validate the model
                scores = cross_validate(
                    model,
                    X,
                    y,
                    scoring=scorer,
                    cv=cv,
                )

            # if not `greater_is_better` the scores will be negaitive
            # multiply by the `sign` to always return positive score
            sign = 1
            if not scorer.greater_is_better:
                sign = -1

            iter_metric = sign * np.mean(scores["test_score"])
            # compare iter metric with the best metric
            if i == 0 or scorer.is_better(iter_metric, best_metric):
                best_metric = iter_metric
                best_C = C
                log.info(f"C={best_C}, metric={best_metric}", msg_type="params")

        self.C = float(best_C)

        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = convert_to_numpy(X_test)

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += fold_model.predict_proba(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def inner_params(self):
        return {
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    @property
    def meta_params(self):
        return {"time_series": self.time_series}

    @property
    def params(self):
        return {**self.inner_params, **self.meta_params}
