import numpy as np
import torch
from catboost import CatBoostClassifier as CBClass
from catboost import CatBoostRegressor as CBReg
from catboost import Pool
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import convert_to_numpy, convert_to_pandas, tune_optuna
from .metrics import get_eval_metric

log = get_logger(__name__)


class CatBoostRegression(BaseModel):
    def __init__(
        self,
        boosting_type="Ordered",
        iterations=None,
        learning_rate=0.03,
        max_leaves=None,
        loss_function=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        od_wait=100,
        rsm=None,
        subsample=None,
        min_data_in_leaf=None,
        grow_policy="SymmetricTree",
        bootstrap_type=None,
        one_hot_max_size=10,
        n_jobs=6,
        random_state=42,
        time_series=False,
        od_type="Iter",
    ):
        self.name = "CatBoostRegression"
        self.cat_features = []

        self.boosting_type = boosting_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.loss_function = loss_function
        self.grow_policy = grow_policy
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.model_size_reg = model_size_reg
        self.od_wait = od_wait
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.subsample = subsample
        self.min_data_in_leaf = min_data_in_leaf
        self.one_hot_max_size = one_hot_max_size
        self.od_type = od_type

        self.thread_count = n_jobs
        self.random_state = random_state
        self.verbose = False
        self.allow_writing_files = False
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.cat_features = categorical_features
        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full(y.shape[0], fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            train_data = Pool(
                X.iloc[train_idx], y[train_idx], cat_features=self.cat_features
            )
            test_data = Pool(
                X.iloc[test_idx], y[test_idx], cat_features=self.cat_features
            )

            # initialize fold model
            fold_model = CBReg(**self.params)

            # fit/predict fold model
            fold_model.fit(train_data, eval_set=test_data)
            oof_preds[test_idx] = fold_model.predict(test_data)

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        # `iterations` is not suggested because it will be corrected by the early stopping
        param_distr = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type",
                [
                    # "Ordered",
                    "Plain",
                ],
            ),
            "depth": trial.suggest_int("depth", 1, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type",
                [
                    "Bernoulli",
                    "MVS",
                ],
            ),
            "grow_policy": trial.suggest_categorical(
                "grow_policy",
                [
                    "SymmetricTree",
                    "Depthwise",
                    "Lossguide",
                ],
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 256),
            "loss_function": trial.suggest_categorical(
                "loss_function",
                [
                    "RMSE",
                    "MAE",
                    "MAPE",
                ],
            ),
            "rsm": trial.suggest_float("rsm", 0.4, 1),
            "subsample": trial.suggest_float("subsample", 0.3, 1),
            "model_size_reg": trial.suggest_float("model_size_reg", 0, 10),
        }

        if param_distr["grow_policy"] == "Lossguide":
            param_distr["max_leaves"] = trial.suggest_int("max_leaves", 10, 512)

        return param_distr

    def get_not_tuned_params(self):
        not_tuned_params = {
            "iterations": 2000,
            "one_hot_max_size": self.one_hot_max_size,
            "learning_rate": self.learning_rate,
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
            "od_type": self.od_type,
            "od_wait": self.od_wait,
            "use_best_model": True,
        }
        if not_tuned_params.get("od_type", "Iter") == "Iter":
            not_tuned_params["od_pval"] = 0
        else:
            not_tuned_params["od_pval"] = 1e-5
        return not_tuned_params

    def objective(self, trial, X, y, scorer):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.get_not_tuned_params()

        cv_metrics = []
        best_num_iterations = []
        for train_idx, test_idx in cv:
            train_data = Pool(
                X.iloc[train_idx], y[train_idx], cat_features=self.cat_features
            )
            test_data = Pool(
                X.iloc[test_idx], y[test_idx], cat_features=self.cat_features
            )

            model = CBReg(**trial_params, **not_tuned_params)

            model.fit(train_data, eval_set=test_data)
            y_pred = model.predict(test_data)

            cv_metrics.append(scorer.score(y[test_idx], y_pred))
            best_num_iterations.append(model.best_iteration_)

        # add `iterations`` to the optuna parameters
        trial.set_user_attr("iterations", round(np.mean(best_num_iterations)))

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

        self.cat_features = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        study = optuna_tune(
            self.name,
            self.objective,
            X=X,
            y=y,
            metric=metric,
            timeout=timeout,
            random_state=self.random_state,
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.iterations = study.best_trial.user_attrs["iterations"]

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = convert_to_pandas(X_test)

        y_pred = np.zeros(X_test.shape[0])
        for fold_model in self.models:
            y_pred += fold_model.predict(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def params(self):
        return {
            **self.get_not_tuned_params(),
            "boosting_type": self.boosting_type,
            "max_leaves": self.max_leaves,
            "loss_function": self.loss_function,
            "grow_policy": self.grow_policy,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "model_size_reg": self.model_size_reg,
            "bootstrap_type": self.bootstrap_type,
            "rsm": self.rsm,
            "subsample": self.subsample,
            "min_data_in_leaf": self.min_data_in_leaf,
        }


class CatBoostClassification(BaseModel):
    def __init__(
        self,
        boosting_type="Plain",
        iterations=2000,
        learning_rate=0.03,
        max_leaves=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        od_wait=100,
        rsm=None,
        subsample=None,
        min_data_in_leaf=None,
        grow_policy="SymmetricTree",
        bootstrap_type=None,
        one_hot_max_size=10,
        auto_class_weights=None,
        thread_count=6,
        random_state=42,
        time_series=False,
        eval_metric=None,
        verbose=False,
        allow_writing_files=False,
        n_jobs=None,
        od_type="Iter",
        od_pval=None,
        task_type=None,
    ):

        self.name = "CatBoostClassification"
        self.cat_features = []

        self.boosting_type = boosting_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.grow_policy = grow_policy
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.model_size_reg = model_size_reg
        self.od_wait = od_wait
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.subsample = subsample
        self.min_data_in_leaf = min_data_in_leaf
        self.one_hot_max_size = one_hot_max_size
        self.auto_class_weights = auto_class_weights

        self.n_jobs = n_jobs
        self.thread_count = thread_count
        self.od_type = od_type
        self.od_pval = od_pval
        self.random_state = random_state
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.time_series = time_series
        self.eval_metric = eval_metric
        self.task_type = task_type

        if self.task_type is None and torch.cuda.is_available():
            task_type = "GPU"
        else:
            task_type = "CPU"

        # correct alias params
        if self.n_jobs is not None:
            self.thread_count = self.n_jobs

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )

        self.models = None
        self.oof_preds = None

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.cat_features = categorical_features
        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            train_data = Pool(
                X.iloc[train_idx], y[train_idx], cat_features=self.cat_features
            )
            test_data = Pool(
                X.iloc[test_idx], y[test_idx], cat_features=self.cat_features
            )

            # initialize fold model
            fold_model = CBClass(**self.inner_params, eval_metric=self.eval_metric)

            # fit/predict fold model
            fold_model.fit(train_data, eval_set=test_data)
            oof_preds[test_idx] = fold_model.predict_proba(test_data)

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        # `iterations` is not suggested because it will be corrected by the early stopping
        param_distr = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type",
                [
                    # "Ordered",
                    "Plain",
                ],
            ),
            "depth": trial.suggest_int("depth", 1, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 200),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type",
                [
                    "Bernoulli",
                    "MVS",
                ],
            ),
            "grow_policy": trial.suggest_categorical(
                "grow_policy",
                [
                    "SymmetricTree",
                    "Depthwise",
                    "Lossguide",
                ],
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 256),
            "rsm": trial.suggest_float("rsm", 0.4, 1),
            "subsample": trial.suggest_float("subsample", 0.4, 1),
            "model_size_reg": trial.suggest_float("model_size_reg", 0, 200),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced", "SqrtBalanced"]
            ),
        }

        if param_distr["grow_policy"] == "Lossguide":
            param_distr["max_leaves"] = trial.suggest_int("max_leaves", 10, 512)

        return param_distr

    def objective(self, trial, X, y, scorer):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params

        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        best_num_iterations = []
        models = []
        for train_idx, test_idx in cv:
            train_data = Pool(
                X.iloc[train_idx], y[train_idx], cat_features=self.cat_features
            )
            test_data = Pool(
                X.iloc[test_idx], y[test_idx], cat_features=self.cat_features
            )

            model = CBClass(
                **trial_params, **not_tuned_params, eval_metric=self.eval_metric
            )

            model.fit(train_data, eval_set=test_data)

            oof_preds[test_idx] = model.predict_proba(test_data)
            best_num_iterations.append(model.best_iteration_)
            models.append(model)

        # add `iterations` as an optuna parameter
        trial.set_user_attr("iterations", round(np.mean(best_num_iterations)))

        # remove possible Nones in oof
        not_none_oof = np.where(np.logical_not(np.isnan(oof_preds[:, 0])))[0]

        if oof_preds.ndim == 2 and oof_preds.shape[1] == 2:
            # binary case
            trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof, 1])
        else:
            trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof])

        # if self.models is None:
        #     # no trials are completed yet
        #     # save tuned models
        #     self.models = models
        #     self.oof_preds = oof_preds

        # elif (trial_metric <= trial.study.best_value and trial.study.direction == 1) or (
        #     trial_metric >= trial.study.best_value and trial.study.direction == 2
        #     ):

        #     # new best
        #     # save tuned models
        #     self.models = models
        #     self.oof_preds = oof_preds

        return trial_metric

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer,
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.cat_features = categorical_features
        self.eval_metric = get_eval_metric(scorer)

        X = convert_to_pandas(X)
        # y = convert_to_numpy(y)
        # y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        study = tune_optuna(
            self.name,
            self.objective,
            X=X,
            y=y,
            scorer=scorer,
            timeout=timeout,
            random_state=self.random_state,
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.iterations = study.best_trial.user_attrs["iterations"]

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = convert_to_pandas(X_test)

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += fold_model.predict_proba(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def not_tuned_params(self):
        not_tuned_params = {
            "iterations": self.iterations,
            "one_hot_max_size": self.one_hot_max_size,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "thread_count": self.thread_count,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
            "od_type": self.od_type,
            "od_wait": self.od_wait,
            "task_type": self.task_type,
        }
        if not_tuned_params.get("od_type", "Iter") == "Iter":
            not_tuned_params["od_pval"] = 0
        else:
            not_tuned_params["od_pval"] = 1e-5
        return not_tuned_params

    @property
    def inner_params(self):
        return {
            "boosting_type": self.boosting_type,
            "max_leaves": self.max_leaves,
            "grow_policy": self.grow_policy,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "model_size_reg": self.model_size_reg,
            "bootstrap_type": self.bootstrap_type,
            "rsm": self.rsm,
            "subsample": self.subsample,
            "min_data_in_leaf": self.min_data_in_leaf,
            "auto_class_weights": self.auto_class_weights,
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
