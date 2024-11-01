import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ...loggers import get_logger
from ..base_model import BaseModel
from ..metrics import MSE
from ..type_hints import FeaturesType, TargetType
from ..utils import LogWhenImproved, convert_to_numpy, convert_to_pandas

log = get_logger(__name__)


class LightGBMRegression(BaseModel):
    def __init__(
        self,
        objective_type="regression",
        boosting="gbdt",
        num_iterations=100,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=20,
        bagging_fraction=1,
        bagging_freq=0,
        feature_fraction=1,
        early_stopping_round=100,
        lambda_l1=0,
        lambda_l2=0,
        min_gain_to_split=0,
        n_jobs=6,
        random_state=42,
        time_series=False,
    ):

        self.name = "LightGBMRegression"

        self.categorical_feature = []

        self.objective_type = objective_type
        self.boosting = boosting
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.feature_fraction = feature_fraction
        self.early_stopping_round = early_stopping_round
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.num_leaves = num_leaves

        self.num_threads = n_jobs
        self.random_state = random_state
        self.verbose = -1
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_feature = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full(y.shape[0], fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            train_data = lgb.Dataset(
                X.iloc[train_idx],
                y[train_idx],
                categorical_feature=self.categorical_feature,
            )
            test_data = lgb.Dataset(
                X.iloc[test_idx],
                y[test_idx],
                categorical_feature=self.categorical_feature,
            )

            # fit/predict fold model
            fold_model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[test_data],
                verbose_eval=False,
            )
            oof_preds[test_idx] = fold_model.predict(X.iloc[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_trial_params(trial):
        param_distr = {
            "objective_type": trial.suggest_categorical(
                "objective_type", ["regression", "regression_l1", "huber"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "num_leaves": trial.suggest_int("num_leaves", 10, 512),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 256),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 20, step=10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 20),
        }

        return param_distr

    def get_not_tuned_params(self):
        not_tuned_params = {
            "boosting": "gbdt",
            "learning_rate": self.learning_rate,
            "num_iterations": 2000,
            "verbose": self.verbose,
            "early_stopping_round": self.early_stopping_round,
            "num_threads": self.num_threads,
            "random_state": self.random_state,
        }
        return not_tuned_params

    def objective(self, trial, X, y, metric):
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

            train_data = lgb.Dataset(
                X.iloc[train_idx],
                y[train_idx],
                categorical_feature=self.categorical_feature,
            )
            test_data = lgb.Dataset(
                X.iloc[test_idx],
                y[test_idx],
                categorical_feature=self.categorical_feature,
            )

            model = lgb.train(
                params={**trial_params, **not_tuned_params},
                train_set=train_data,
                valid_sets=[test_data],
                verbose_eval=False,
            )
            y_pred = model.predict(X.iloc[test_idx])

            cv_metrics.append(metric(y[test_idx], y_pred))
            best_num_iterations.append(model.best_iteration)

        # add `num_iterations` to the optuna parameters
        trial.set_user_attr("num_iterations", round(np.mean(best_num_iterations)))

        return np.mean(cv_metrics)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        metric=MSE(),
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.categorical_feature = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        # seed sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # optimize parameters
        study = optuna.create_study(
            study_name=self.name,
            direction="maximize" if metric.greater_is_better else "minimize",
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self.objective(trial, X, y, metric),
            timeout=timeout,
            n_jobs=1,
            callbacks=[LogWhenImproved()],
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.num_iterations = study.best_trial.user_attrs["num_iterations"]

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
            "objective_type": self.objective_type,
            "boosting": self.boosting,
            "num_iterations": self.num_iterations,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "feature_fraction": self.feature_fraction,
            "early_stopping_round": self.early_stopping_round,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "num_threads": self.num_threads,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


class LightGBMClassification(BaseModel):
    def __init__(
        self,
        objective_type="binary",
        boosting="gbdt",
        num_iterations=100,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=20,
        bagging_fraction=1,
        bagging_freq=0,
        feature_fraction=1,
        early_stopping_round=100,
        lambda_l1=0,
        lambda_l2=0,
        min_gain_to_split=0,
        is_unbalance=False,
        class_weight=None,
        n_jobs=6,
        random_state=42,
        time_series=False,
    ):

        self.name = "LightGBMClassification"

        self.categorical_feature = []

        self.objective_type = objective_type
        self.boosting = boosting
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.feature_fraction = feature_fraction
        self.early_stopping_round = early_stopping_round
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.num_leaves = num_leaves
        self.is_unbalance = is_unbalance
        self.class_weight = class_weight

        self.num_threads = n_jobs
        self.random_state = random_state
        self.verbose = -1
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_feature = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        # correct objective based on the number of classes
        if self.n_classes > 2:
            self.objective_type = "multiclass"

        cv = self.kf.split(X, y)

        self.models = []
        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            params = self.params

            # BUG LightGBM produces very annoying alias warning.
            # Temp fix is to rename all the input parameters.
            # In the recent versions of lightgbm the issue is solved.
            params["colsample_bytree"] = params.pop("feature_fraction")
            params["reg_alpha"] = params.pop("lambda_l1")
            params["subsample"] = params.pop("bagging_fraction")
            params["min_child_samples"] = params.pop("min_data_in_leaf")
            params["min_split_gain"] = params.pop("min_gain_to_split")
            params["subsample_freq"] = params.pop("bagging_freq")
            params["reg_lambda"] = params.pop("lambda_l2")

            params["n_jobs"] = params.pop("num_threads")
            params["boosting_type"] = params.pop("boosting")

            # fit/predict fold model
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
                early_stopping_rounds=self.early_stopping_round,
                categorical_feature=self.categorical_feature,
            )
            oof_preds[test_idx] = fold_model.predict_proba(X.iloc[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    def get_trial_params(self, trial):
        param_distr = {
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "num_leaves": trial.suggest_int("num_leaves", 10, 512),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 256),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 20, step=10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 20),
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        }

        return param_distr

    def get_not_tuned_params(self):
        not_tuned_params = {
            "boosting": "gbdt",
            "num_iterations": 2000,
            "learning_rate": self.learning_rate,
            "objective_type": self.objective_type,
            "num_classes": 1 if self.n_classes == 2 else self.n_classes,
            "verbose": self.verbose,
            "early_stopping_round": self.early_stopping_round,
            "num_threads": self.num_threads,
            "random_state": self.random_state,
        }
        return not_tuned_params

    def objective(self, trial, X, y, metric):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.get_not_tuned_params()

        # BUG LightGBM produces very annoying alias warning.
        # Temp fix is to rename all the input parameters.
        # In the recent versions of lightgbm the issue is solved.
        trial_params["colsample_bytree"] = trial_params.pop("feature_fraction")
        trial_params["reg_alpha"] = trial_params.pop("lambda_l1")
        trial_params["subsample"] = trial_params.pop("bagging_fraction")
        trial_params["min_child_samples"] = trial_params.pop("min_data_in_leaf")
        trial_params["min_split_gain"] = trial_params.pop("min_gain_to_split")
        trial_params["subsample_freq"] = trial_params.pop("bagging_freq")
        trial_params["reg_lambda"] = trial_params.pop("lambda_l2")

        not_tuned_params["n_jobs"] = not_tuned_params.pop("num_threads")
        not_tuned_params["boosting_type"] = not_tuned_params.pop("boosting")

        # add parameter `class_weight` for multiclass only
        if self.n_classes > 2:
            trial_params["class_weight"] = trial.suggest_categorical(
                "class_weight", ["balanced", None]
            )

        cv_metrics = []
        best_num_iterations = []
        for train_idx, test_idx in cv:
            # train_data = lgb.Dataset(
            #     X[train_idx], y[train_idx], categorical_feature=self.categorical_feature
            # )
            # test_data = lgb.Dataset(
            #     X[test_idx], y[test_idx], categorical_feature=self.categorical_feature
            # )

            # model = lgb.train(
            #     params={**trial_params, **not_tuned_params},
            #     train_set=train_data,
            #     valid_sets=[test_data],
            #     verbose_eval=False
            # )

            # Here I decided to use sklearn API
            # because `lgb.train` does not contain parameter `class_weight`
            # that drastically improves the performance of models
            model = lgb.LGBMClassifier(**trial_params, **not_tuned_params)
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
                early_stopping_rounds=not_tuned_params["early_stopping_round"],
                categorical_feature=self.categorical_feature,
            )

            y_pred = model.predict_proba(X.iloc[test_idx])

            cv_metrics.append(metric(y[test_idx], y_pred))
            best_num_iterations.append(model.best_iteration_)

        # add `num_iterations` to the optuna parameters
        trial.set_user_attr("num_iterations", round(np.mean(best_num_iterations)))

        return np.mean(cv_metrics)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        metric=MSE(),
        timeout=60,
        categorical_features=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.categorical_feature = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        if self.n_classes > 2:
            self.objective_type = "multiclass"

        # seed sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # optimize parameters
        study = optuna.create_study(
            study_name=self.name,
            direction="maximize" if metric.greater_is_better else "minimize",
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self.objective(trial, X, y, metric),
            timeout=timeout,
            n_jobs=1,
            callbacks=[LogWhenImproved()],
        )

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self, key, val)
        self.num_iterations = study.best_trial.user_attrs["num_iterations"]

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
    def params(self):
        params = {
            "objective_type": self.objective_type,
            "boosting": self.boosting,
            "num_iterations": self.num_iterations,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "feature_fraction": self.feature_fraction,
            "early_stopping_round": self.early_stopping_round,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "num_threads": self.num_threads,
            "random_state": self.random_state,
            "is_unbalance": self.is_unbalance,
            "num_classes": 1 if self.n_classes == 2 else self.n_classes,
            "verbose": self.verbose,
        }

        # add parameter `class_weight` for multiclass only
        if self.n_classes > 2:
            params["class_weight"] = self.class_weight

        return params
