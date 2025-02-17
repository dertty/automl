import numpy as np
from catboost import CatBoostClassifier as CBClass
from catboost import CatBoostRegressor as CBReg
from catboost import Pool

from typing import Any, Optional, List, Dict, Union
from ...loggers import get_logger
from ..base import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import tune_optuna
from ..utils.model_utils import get_splitter, get_epmty_array


log = get_logger(__name__)


class CatBoostBase(BaseModel):   
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
        time_series: bool = False,
        verbose: bool = False,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            )
        
        if self.model_type == 'classification':
            self.name: str = "CatBoostClassification"
            self.model = CBClass
            self.model_predict_func_name: str = 'predict_proba'
        elif self.model_type == 'regression':
            self.name = "CatBoostRegression"
            self.model = CBReg
            self.model_predict_func_name = 'predict'
                
        # model params
        self.thread_count = kwargs.pop('thread_count', self.n_jobs)
        self.task_type = kwargs.pop('task_type', self.device_type).upper()
        self.verbose = kwargs.pop('verbose', None) or kwargs.pop('verbose_eval', self.verbose)
        # other model params
        self.iterations: int = kwargs.pop('num_iterations', 2_000)
        self.iterations = kwargs.pop('iterations', None) or kwargs.pop('n_iterations', self.iterations)
        self.max_iterations = self.iterations
        self.od_type = kwargs.pop('od_type', 'Iter')
        self.od_wait = kwargs.pop('od_wait', 100)
        self.od_pval = None if self.od_type == "Iter" else kwargs.pop('od_pval', 1e-5)
        self.logging_level = kwargs.pop('logging_level', 'Silent')
        # fit params
        self.cat_features: List[str] = []
        self.models: Optional[List[Union[CBClass, CBReg]]] = None
        self.n_classes: Optional[int] = None
        
        self._not_inner_model_params += [
            'model', 'model_predict_func_name', 'n_classes', 
            'n_jobs', 'eval_metric', 'device_type', 'verbose',]
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def _prepare(self, X: FeaturesType, y: TargetType = None, categorical_feature: Optional[List[Union[str, int]]] = None):
        X, y = self._prepare_data(X, y, categorical_feature)
        self.cat_features = self.categorical_feature
        if self.categorical_feature:
            filtered_columns = X.select_dtypes(exclude=['object', 'int']).columns
            cast_columns = list(set(self.categorical_feature) & set(filtered_columns))
            X[cast_columns] = X[cast_columns].astype(str)
        if y is not None:
            if self.model_type == "classification":
                self.n_classes = np.unique(y).shape[0]
            return X, y
        return X

    def fit_fold(
        self, 
        X_train: FeaturesType, y_train: TargetType,
        X_test: FeaturesType, y_test: TargetType,
        inner_params: Dict[Any, Any] = {}
        ):
        train_data = Pool(
            X_train, y_train, 
            cat_features=self.categorical_feature)
        test_data = Pool(
            X_test, y_test, 
            cat_features=self.categorical_feature)
        
        fold_model = self.model(**inner_params)
        fold_model.fit(train_data, eval_set=test_data, verbose=self.verbose)
        
        best_score_dict: Dict[str, float] = fold_model.get_best_score()['validation']
        for _, value in best_score_dict.items():
            best_score: float = value
        return fold_model, best_score
        
    def fit(self, X: FeaturesType, y: TargetType, categorical_feature: Optional[List[Union[str, int]]] = None):
        log.info(f"Fitting {self.name}", msg_type="start")

        X, y = self._prepare(X, y, categorical_feature)
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.random_state)
        cv = kf.split(X, y)
        self.models = []
        oof_preds = get_epmty_array(y.shape[0], self.n_classes)
        for i, (train_idx, test_idx) in enumerate(cv):
            # log.info(f"{self.name} fold {i}", msg_type="fit")
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            fold_model, _ = self.fit_fold(
                X_train, y_train, 
                X_test, y_test,
                inner_params=self.inner_params)
            test_data = Pool(X_test, cat_features=self.categorical_feature)
            oof_preds[test_idx] = getattr(fold_model, self.model_predict_func_name)(test_data)
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_base_trial_params(trial):
        # `iterations` is not suggested because it will be corrected by the early stopping
        default_param_distr = {
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
        }

        if default_param_distr["grow_policy"] == "Lossguide":
            default_param_distr["max_leaves"] = trial.suggest_int("max_leaves", 10, 512)

        return default_param_distr
        
    def optuna_objective(self, trial, X: FeaturesType, y: TargetType):
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.random_state)
        cv = kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params
        not_tuned_params['iterations'] = self.max_iterations
        inner_params = {**not_tuned_params, **trial_params}

        best_num_iterations = []
        scores = []
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            fold_model, score = self.fit_fold(
                X_train, y_train,
                X_test, y_test,
                inner_params=inner_params)
            scores.append(score)
            # oof_preds[test_idx] = getattr(fold_model, self.model_predict_func_name)(test_data)
            best_num_iterations.append(fold_model.best_iteration_)

        # add `iterations` as an optuna parameter
        trial.set_user_attr("iterations", max(1, round(np.mean(best_num_iterations))))
        return np.mean(scores)
        # # remove possible Nones in oof
        # not_none_oof = np.where(np.logical_not(np.isnan(oof_preds[:, 0])))[0]

        # if self.model_type == 'classification' and self.n_classes == 2:
        #     # binary case
        #     trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof, 1])
        # else:
        #     trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof])

        # return trial_metric

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        timeout: int=60,
        categorical_feature: Optional[List[str]] = None,
    ):
        log.info(f"Tuning {self.name}", msg_type="start")
        
        from .metrics import METRICS_GREATER_IS_BETTER
        
        if isinstance(self.eval_metric, str):
            greater_is_better = self.get_str_greater_is_better(
                self.model_type, 
                self.eval_metric, 
                METRICS_GREATER_IS_BETTER)
        elif self.eval_metric is not None:
            greater_is_better = self.eval_metric.is_max_optimal()
        else:
            greater_is_better = None

        X, y = self._prepare(X, y, categorical_feature)
        study = tune_optuna(
            self.name,
            self.optuna_objective,
            X=X, y=y,
            greater_is_better=greater_is_better,
            timeout=timeout,
            random_state=self.random_state,)
        self.best_params = study.best_params
        self.best_params['iterations'] = study.best_trial.user_attrs["iterations"]

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X: FeaturesType):
        """Predict on one dataset. Average all fold models"""
        X = self._prepare(X, categorical_feature=self.categorical_feature)
        
        y_pred = np.zeros((X.shape[0], self.n_classes)) \
            if self.n_classes else np.zeros((X.shape[0],))
        for fold_model in self.models:
            y_pred += getattr(fold_model, self.model_predict_func_name)(X)
        y_pred = y_pred / len(self.models)
        
        return y_pred

    @property
    def not_tuned_params(self) -> dict:
        not_tuned_params = {
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            # "verbose": self.verbose,
            'logging_level': self.logging_level,
            "task_type": self.task_type,
            "od_type": self.od_type,
            "od_wait": self.od_wait,
            "od_pval": self.od_pval,
            
        }
        return {key: value for key, value in not_tuned_params.items() if value}

    @property
    def inner_params(self) -> dict:
        return {
            'iterations': self.iterations,
            **self.not_tuned_params,
            **{key: value for key, value in self.__dict__.items() if key not in self._not_inner_model_params},
            **self.best_params,
        }

class CatBoostClassification(CatBoostBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: bool = False,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type="classification",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
        )
    
    @staticmethod   
    def get_trial_params(trial):
        params = CatBoostBase.get_base_trial_params(trial)
        params["auto_class_weights"] = trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", None])
        return params


class CatBoostRegression(CatBoostBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: bool = False,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type="regression",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
        )
    
    @staticmethod
    def get_trial_params(trial):
        params = CatBoostBase.get_base_trial_params(trial)
        params.update({
            "loss_function": trial.suggest_categorical(
                "loss_function", ["RMSE", "MAE", "MAPE",]
            ),
        })

        return params
