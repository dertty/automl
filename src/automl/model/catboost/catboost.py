import tempfile
import shutil
from pathlib import Path
import numpy as np

from catboost import CatBoostClassifier as CBClass
from catboost import CatBoostRegressor as CBReg
from catboost import Pool

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ...loggers import get_logger
from ..base_model import BaseModel
from ..metrics import MSE
from ..type_hints import FeaturesType, TargetType
from ..utils import optuna_tune, convert_to_numpy, convert_to_pandas

log = get_logger(__name__)


class CatBoostRegression(BaseModel):
    def __init__(
        self,
        boosting_type="Ordered",
        iterations=2000,
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
        od_type='Iter',
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
            
        self.tmp_dir = Path(tempfile.mkdtemp())

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
            "iterations": self.iterations,
            "one_hot_max_size": self.one_hot_max_size,
            "learning_rate": self.learning_rate,
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
            "od_type": self.od_type,
            "od_wait": self.od_wait,
            'use_best_model': True,
        }
        if not_tuned_params.get('od_type', 'Iter') == 'Iter':
            not_tuned_params["od_pval"] = 0
        else:
            not_tuned_params["od_pval"] = 1e-5
        return not_tuned_params

    def objective(self, trial, X, y, metric, **kwargs):
        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.get_not_tuned_params()

        cv_metrics = []
        best_num_iterations = []
        if 'folds' in kwargs:
            for train_idx_path, test_idx_path, train_idx, test_idx in kwargs['folds']:
                if train_idx_path.exists():
                    train_data = Pool(data='quantized://' + str(train_idx_path))
                else:
                    train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
                    
                test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)
                if test_idx_path.exists():
                    eval_data = Pool(data='quantized://' + str(test_idx_path))
                else:
                    eval_data = test_data
                
                model = CBClass(**trial_params, **not_tuned_params)

                model.fit(train_data, eval_set=eval_data)
                y_pred = model.predict_proba(test_data)

                cv_metrics.append(metric(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)
        else:
            for train_idx, test_idx in self.kf.split(X, y):
                train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
                test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)

                model = CBClass(**trial_params, **not_tuned_params)

                model.fit(train_data, eval_set=test_data)
                y_pred = model.predict_proba(test_data)

                cv_metrics.append(metric(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)

        # add `iterations`` to the optuna parameters
        trial.set_user_attr("iterations", round(np.mean(best_num_iterations)))

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

        self.cat_features = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        cv = self.kf.split(X, y)
        
        folds = []
        for i, (train_idx, test_idx) in enumerate(cv):
            train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
            test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)
            train_data.quantize()
            test_data.quantize()
            
            train_data.save(self.tmp_dir / f'train_data_{i}')
            test_data.save(self.tmp_dir / f'test_data{i}')
            
            folds.append([self.tmp_dir / f'train_data_{i}', self.tmp_dir / f'test_data{i}', train_idx, test_idx])
            
        study = optuna_tune(
            name=self.name, 
            objective=self.objective, 
            X=X, y=y, 
            metric=metric, timeout=timeout, random_state=self.random_state, 
            folds=folds)

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
        
    def __del__(self):
        # Удаляем временную папку и её содержимое
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)


class CatBoostClassification(BaseModel):
    def __init__(
        self,
        boosting_type="Ordered",
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
        n_jobs=6,
        random_state=42,
        time_series=False,
        od_type='Iter',
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
        self.od_type = od_type
        
        self.thread_count = n_jobs
        self.random_state = random_state
        self.verbose = False
        self.allow_writing_files = False
        self.time_series = time_series

        if self.time_series:
            self.kf = TimeSeriesSplit(n_splits=5)
        else:
            self.kf = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )
        
        self.tmp_dir = Path(tempfile.mkdtemp())

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
            fold_model = CBClass(**self.params)

            # fit/predict fold model
            fold_model.fit(train_data, eval_set=test_data)
            oof_preds[test_idx] = fold_model.predict_proba(test_data)

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds
    
    @staticmethod
    def get_trial_params(trial, mode: str='precision', dataset_shape: tuple[int, int] = (0, 0)):
        # `iterations` is not suggested because it will be corrected by the early stopping
        param_distr = {}
        match mode:
            case 'fast':
                param_distr['boosting_type'] = trial.suggest_categorical("boosting_type", ["Plain",])
                param_distr["bootstrap_type"] = trial.suggest_categorical("bootstrap_type", ["MVS",])
                param_distr["grow_policy"] = trial.suggest_categorical("grow_policy", ["Depthwise"])
                
                if dataset_shape[1] > 1_000:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.3, 0.3, step=0.01)
                elif dataset_shape[1] > 500:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.4, 0.4, step=0.01)
                elif dataset_shape[1] > 100:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.5, 0.5, step=0.01)
                elif dataset_shape[1] > 50:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.6, 0.6, step=0.01)
                elif dataset_shape[1] > 20:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.8, 0.8, step=0.01)
                else:
                    param_distr["rsm"] = trial.suggest_float("rsm", 1., 1., step=0.01)

                if dataset_shape[0] > 1_000_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.5, 0.5, step=0.01)
                elif dataset_shape[0] > 100_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.6, 0.6, step=0.01)
                elif dataset_shape[0] > 10_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.7, 0.7, step=0.01)
                elif dataset_shape[0] > 1_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.8, 0.8, step=0.01)
                else:
                    param_distr["subsample"] = trial.suggest_float("subsample", 1, 1, step=0.01)
            case 'balanced':
                param_distr['boosting_type'] = trial.suggest_categorical("boosting_type", ["Plain",])
                param_distr["bootstrap_type"] = trial.suggest_categorical("bootstrap_type", ["MVS",])
                param_distr["grow_policy"] = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide",])
                
                if dataset_shape[1] > 500:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.01, 0.6, step=0.01)
                elif dataset_shape[1] > 100:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.2, 0.8, step=0.01)
                elif dataset_shape[1] > 50:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.3, 1., step=0.01)
                elif dataset_shape[1] > 20:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.4, 1., step=0.01)
                else:
                    param_distr["rsm"] = trial.suggest_float("rsm", 0.8, 1., step=0.01)

                if dataset_shape[0] > 1_000_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.01, 0.5, step=0.01)
                elif dataset_shape[0] > 500_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.2, 0.8, step=0.01)
                elif dataset_shape[0] > 100_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.2, 1, step=0.01)
                elif dataset_shape[0] > 10_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.4, 1, step=0.01)
                elif dataset_shape[0] > 1_000:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.6, 1, step=0.01)
                else:
                    param_distr["subsample"] = trial.suggest_float("subsample", 0.8, 1, step=0.01)
            case _:
                param_distr['boosting_type'] = trial.suggest_categorical("boosting_type", ["Ordered", "Plain",])
                param_distr["bootstrap_type"] = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "MVS",])
                if param_distr["boosting_type"] == "Ordered":
                    param_distr["grow_policy"] = trial.suggest_categorical("grow_policy", ["SymmetricTree",])
                else:
                    param_distr["grow_policy"] = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide",])
                param_distr["rsm"] = trial.suggest_float("rsm", 0.4, 1.)
                param_distr["subsample"] = trial.suggest_float("subsample", 0.4, 1)
                
        param_distr['depth'] = trial.suggest_int("depth", 1, 16)
        param_distr['l2_leaf_reg'] = trial.suggest_float("l2_leaf_reg", 0, 200)
        param_distr['auto_class_weights'] = trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced",])
        param_distr['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 1, 256)
        param_distr['model_size_reg'] = trial.suggest_float("model_size_reg", 0, 200)
        if param_distr["grow_policy"] == "Lossguide":
            param_distr["max_leaves"] = trial.suggest_int("max_leaves", 10, 512)
        return param_distr

    def get_not_tuned_params(self):
        not_tuned_params = {
            "iterations": self.iterations,
            "one_hot_max_size": self.one_hot_max_size,
            "learning_rate": self.learning_rate,
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
            "od_type": self.od_type,
            "od_wait": self.od_wait,
            'use_best_model': True,
        }
        if not_tuned_params.get('od_type', 'Iter') == 'Iter':
            not_tuned_params["od_pval"] = 0
        else:
            not_tuned_params["od_pval"] = 1e-5
        return not_tuned_params

    def objective(self, trial, X, y, metric, mode='fast', **kwargs):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial, mode=mode)
        not_tuned_params = self.get_not_tuned_params()

        cv_metrics = []
        best_num_iterations = []
        if 'folds' in kwargs:
            for train_idx_path, test_idx_path, train_idx, test_idx in kwargs['folds']:
                if train_idx_path.exists():
                    train_data = Pool(data='quantized://' + str(train_idx_path))
                else:
                    train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
                    
                test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)
                if test_idx_path.exists():
                    eval_data = Pool(data='quantized://' + str(test_idx_path))
                else:
                    eval_data = test_data
                
                model = CBClass(**trial_params, **not_tuned_params)

                model.fit(train_data, eval_set=eval_data)
                y_pred = model.predict_proba(test_data)

                cv_metrics.append(metric(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)
        else:
            for train_idx, test_idx in self.kf.split(X, y):
                train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
                test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)

                model = CBClass(**trial_params, **not_tuned_params)

                model.fit(train_data, eval_set=test_data)
                y_pred = model.predict_proba(test_data)

                cv_metrics.append(metric(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)

        # add `iterations` as an optuna parameters
        trial.set_user_attr("iterations", round(np.mean(best_num_iterations)))

        return np.mean(cv_metrics)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        metric=MSE(),
        timeout=60,
        categorical_features=[],
        mode='fast',
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.cat_features = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])

        cv = self.kf.split(X, y)
        folds = []
        for i, (train_idx, test_idx) in enumerate(cv):
            train_data = Pool(X.iloc[train_idx], y[train_idx], cat_features=self.cat_features)
            test_data = Pool(X.iloc[test_idx], y[test_idx], cat_features=self.cat_features)
            train_data.quantize()
            test_data.quantize()
            
            train_data.save(self.tmp_dir / f'train_data_{i}')
            test_data.save(self.tmp_dir / f'test_data{i}')
            
            folds.append([self.tmp_dir / f'train_data_{i}', self.tmp_dir / f'test_data{i}', train_idx, test_idx])
            
        study = optuna_tune(
            name=self.name, 
            objective=self.objective, 
            X=X, y=y, 
            metric=metric, 
            timeout=timeout, 
            random_state=self.random_state, 
            mode=mode,
            folds=folds)

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
    def params(self):
        return {
            **self.get_not_tuned_params(),
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
        }

    def __del__(self):
        # Удаляем временную папку и её содержимое
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)