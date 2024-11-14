import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier as RFClassSklearn
from sklearn.ensemble import RandomForestRegressor as RFRegSklearn
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)

from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import optuna_tune, convert_to_numpy

log = get_logger(__name__)


class RandomForestRegression(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=6,
        random_state=42,
        verbose=0,
        max_samples=None,
        time_series=False,
    ):

        self.name = "RandomForestRegression"

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.time_series = time_series

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
        cv = self.kf.split(X, y)
        oof_preds = np.full(y.shape[0], fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            # initialize fold model
            fold_model = RFRegSklearn(**self.params)

            # fit/predict fold model
            fold_model.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = fold_model.predict(X[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        param_distr = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_float("min_samples_split", 0, 0.2),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0, 0.2),
            "bootstrap": True,
            "max_features": trial.suggest_float("max_features", 0.1, 1),
            "criterion": trial.suggest_categorical(
                "criterion",
                [
                    "squared_error",
                    "absolute_error",
                    "friedman_mse",
                ],
            ),
            "oob_score": trial.suggest_categorical("oob_score", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.01, 1),
        }

        return param_distr

    def objective(self, trial, X, y, scorer):
        cv = self.kf.split(X, y)
        trial_params = self.get_trial_params(trial)

        model = RFRegSklearn(
            **trial_params, n_jobs=self.n_jobs, random_state=self.random_state
        )
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

        return sign * np.mean(scores["test_score"])

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        study = optuna_tune(self.name, self.objective, X=X, y=y, metric=metric, timeout=timeout, random_state=self.random_state)

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
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
        return {
            **self.get_not_tuned_params(),
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "max_samples": self.max_samples,
        }


class RandomForestClassification(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=6,
        random_state=42,
        verbose=0,
        max_samples=None,
        class_weight="balanced",
        time_series=False,
    ):

        self.name = "RandomForestClassification"

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
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
            fold_model = RFClassSklearn(**self.inner_params)

            # fit/predict fold model
            fold_model.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = fold_model.predict_proba(X[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        param_distr = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_float("min_samples_split", 0, 0.2),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0, 0.2),
            "bootstrap": True,
            "max_features": trial.suggest_float("max_features", 0.1, 1),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy", "log_loss"],
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            "oob_score": trial.suggest_categorical("oob_score", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.01, 1),
        }

        return param_distr

    def objective(self, trial, X, y, scorer):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params

        model = RFClassSklearn(**trial_params, **not_tuned_params)

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

        return sign * np.mean(scores["test_score"])

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        X = convert_to_numpy(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        study = optuna_tune(self.name, self.objective, X=X, y=y, metric=metric, timeout=timeout, random_state=self.random_state)

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = X_test.to_numpy()

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += fold_model.predict_proba(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def not_tuned_params(self):
        return {
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    @property
    def inner_params(self):
        return {
            **self.get_not_tuned_params(),
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "max_samples": self.max_samples,
            "class_weight": self.class_weight,
            **self.not_tuned_params,
        }

    @property
    def meta_params(self):
        return {"time_series": self.time_series}

    @property
    def params(self):
        return {**self.inner_params, **self.meta_params}
