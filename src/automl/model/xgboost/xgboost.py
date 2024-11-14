from copy import deepcopy

import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier as XGBClass
from xgboost import XGBRegressor as XGBReg

from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import LogWhenImproved, convert_to_numpy, convert_to_pandas
from .metrics import get_eval_metric

log = get_logger(__name__)


class XGBRegression(BaseModel):
    def __init__(
        self,
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=None,
        max_leaves=None,
        grow_policy="lossguide",
        gamma=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_lambda=1,
        reg_alpha=0,
        early_stopping_rounds=100,
        enable_categorical=True,
        max_cat_to_onehot=5,
        n_jobs=6,
        random_state=42,
        time_series=False,
    ):

        self.name = "XGBRegression"

        self.categorical_features = []
        self.enable_categorical = enable_categorical
        self.max_cat_to_onehot = max_cat_to_onehot

        self.objective_type = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grow_policy = grow_policy
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = 0
        self.early_stopping_rounds = early_stopping_rounds
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_features = categorical_features

        X = deepcopy(convert_to_pandas(X))
        X.loc[:, self.categorical_features] = X[self.categorical_features].astype(
            "category"
        )
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full(y.shape[0], fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            fold_model = XGBReg(**self.params)

            # fit/predict fold model
            fold_model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
            )

            oof_preds[test_idx] = fold_model.predict(X.iloc[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        param_distr = {
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "max_leaves": trial.suggest_int("max_leaves", 10, 512),
            "gamma": trial.suggest_float("gamma", 0, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
        }

        return param_distr

    def get_not_tuned_params(self):
        not_tuned_params = {
            "objective": self.objective_type,
            "learning_rate": self.learning_rate,
            "n_estimators": 2000,
            "verbosity": self.verbosity,
            "early_stopping_rounds": self.early_stopping_rounds,
            "enable_categorical": self.enable_categorical,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }
        return not_tuned_params

    def objective(self, trial, X, y, scorer):
        """
        Perform cross-validation to evaluate the model.
        Mean test score is returned.
        """
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.get_not_tuned_params()

        cv_metrics = []
        best_num_iterations = []
        for train_idx, test_idx in cv:

            model = XGBReg(**trial_params, **not_tuned_params)
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
            )
            y_pred = model.predict(X.iloc[test_idx])

            cv_metrics.append(scorer.score(y[test_idx], y_pred))
            best_num_iterations.append(model.best_iteration)

        # add `n_estimators` to the optuna parameters
        trial.set_user_attr("n_estimators", round(np.mean(best_num_iterations)))

        return np.mean(cv_metrics)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.categorical_features = categorical_features

        X = deepcopy(convert_to_pandas(X))
        X.loc[:, self.categorical_features] = X[self.categorical_features].astype(
            "category"
        )
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        # seed sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # optimize parameters
        study = optuna.create_study(
            study_name=self.name,
            direction="maximize" if scorer.greater_is_better else "minimize",
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self.objective(trial, X, y, scorer.score),
            timeout=timeout,
            n_jobs=1,
            callbacks=[LogWhenImproved()],
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.n_estimators = study.best_trial.user_attrs["n_estimators"]

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = deepcopy(convert_to_pandas(X_test))
        X_test.loc[:, self.categorical_features] = X_test[
            self.categorical_features
        ].astype("category")

        y_pred = np.zeros(X_test.shape[0])
        for fold_model in self.models:
            y_pred += fold_model.predict(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def params(self):
        return {
            "objective": self.objective_type,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "grow_policy": self.grow_policy,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "enable_categorical": self.enable_categorical,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "early_stopping_rounds": self.early_stopping_rounds,
        }


class XGBClassification(BaseModel):
    def __init__(
        self,
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.03,
        max_leaves=None,
        max_depth=None,
        grow_policy="lossguide",
        gamma=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_lambda=1,
        reg_alpha=0,
        early_stopping_rounds=100,
        enable_categorical=True,
        max_cat_to_onehot=5,
        n_jobs=6,
        random_state=42,
        verbosity=0,
        class_weight=None,
        time_series=False,
        eval_metric=None,
    ):

        self.name = "XGBClassification"

        self.categorical_features = []
        self.enable_categorical = enable_categorical
        self.max_cat_to_onehot = max_cat_to_onehot

        self.objective_type = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grow_policy = grow_policy
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.class_weight = class_weight

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.time_series = time_series
        self.eval_metric = eval_metric

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_features = categorical_features

        X = deepcopy(convert_to_pandas(X))
        X.loc[:, self.categorical_features] = X[self.categorical_features].astype(
            "category"
        )
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        # correct `objective_type` based on the number of classes
        if self.n_classes > 2:
            self.objective_type = "multi:softmax"

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            inner_params = self.inner_params

            sample_weight = compute_sample_weight(class_weight=None, y=y[train_idx])

            # add class weights
            # in binary case -> `scale_pos_weight`
            # in multiclass case -> `sample_weight`
            if self.n_classes == 2 and self.class_weight == "balanced":
                # binary case
                class_count = np.bincount(y[train_idx].astype(int))
                inner_params["scale_pos_weight"] = class_count[0] / class_count[1]
            elif self.n_classes > 2 and self.class_weight == "balanced":
                # multiclass case
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y[train_idx]
                )

            fold_model = XGBClass(**inner_params, eval_metric=self.eval_metric)

            # fit/predict fold model
            fold_model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
                sample_weight=sample_weight,
            )

            oof_preds[test_idx] = fold_model.predict_proba(X.iloc[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        param_distr = {
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "max_leaves": trial.suggest_int("max_leaves", 10, 512),
            "gamma": trial.suggest_float("gamma", 0, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }

        return param_distr

    def objective(self, trial, X, y, scorer):
        """
        Perform cross-validation to evaluate the model.
        Mean test score is returned.
        """
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params

        class_weight = trial_params.pop("class_weight")

        cv_metrics = []
        best_num_iterations = []
        for train_idx, test_idx in cv:
            sample_weight = compute_sample_weight(class_weight=None, y=y[train_idx])

            # add class weights
            # in binary case -> `scale_pos_weight`
            # in multiclass case -> `sample_weight`
            if self.n_classes == 2 and class_weight == "balanced":
                # binary case
                class_count = np.bincount(y[train_idx].astype(int))
                not_tuned_params["scale_pos_weight"] = class_count[0] / class_count[1]
            elif self.n_classes > 2 and class_weight == "balanced":
                # multiclass case
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y[train_idx]
                )

            model = XGBClass(
                **trial_params, **not_tuned_params, eval_metric=self.eval_metric
            )
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
                sample_weight=sample_weight,
            )

            y_pred = model.predict_proba(X.iloc[test_idx])

            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]

            cv_metrics.append(scorer.score(y[test_idx], y_pred))
            best_num_iterations.append(model.best_iteration)

        # add `n_estimators` to the optuna parameters
        trial.set_user_attr("n_estimators", round(np.mean(best_num_iterations)))

        return np.mean(cv_metrics)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.categorical_features = categorical_features
        self.eval_metric = get_eval_metric(scorer)

        X = deepcopy(convert_to_pandas(X))
        X.loc[:, self.categorical_features] = X[self.categorical_features].astype(
            "category"
        )
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        # correct `objective_type` based on the number of classes
        if self.n_classes > 2:
            self.objective_type = "multi:softmax"

        # seed sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # optimize parameters
        study = optuna.create_study(
            study_name=self.name,
            direction="maximize" if scorer.greater_is_better else "minimize",
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self.objective(trial, X, y, scorer),
            timeout=timeout,
            n_jobs=1,
            callbacks=[LogWhenImproved()],
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.n_estimators = study.best_trial.user_attrs["n_estimators"]

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = deepcopy(convert_to_pandas(X_test))
        X_test.loc[:, self.categorical_features] = X_test[
            self.categorical_features
        ].astype("category")

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += fold_model.predict_proba(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def not_tuned_params(self):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "verbosity": self.verbosity,
            "early_stopping_rounds": self.early_stopping_rounds,
            "enable_categorical": self.enable_categorical,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    @property
    def inner_params(self):
        return {
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "grow_policy": self.grow_policy,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "class_weight": self.class_weight,
            **self.not_tuned_params,
        }

    @property
    def meta_params(self):
        return {
            "eval_metric": (
                self.eval_metric
                if isinstance(self.eval_metric, str) or self.eval_metric is None
                else "custom_metric"
            ),
            "time_series": self.time_series,
        }

    @property
    def params(self):
        return {
            **self.inner_params,
            **self.meta_params,
        }
