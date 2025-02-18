import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer

from typing import Any, Optional, List, Dict, Union
from ...loggers import get_logger
from ..base import BaseModel
from ..type_hints import FeaturesType, TargetType, ScorerType
from ..utils import tune_optuna
from ..utils.model_utils import get_splitter, get_epmty_array


log = get_logger(__name__)


class SKBase(BaseModel):
    def __init__(
        self,
        model_type: str,
        model: BaseEstimator,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, ScorerType]] = None,
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
        self.model = model
        self.name: str = self.model.__class__.__name__
        if self.model_type == 'classification':
            self.model_predict_func_name = 'predict_proba'
        elif self.model_type == 'regression':
            self.model_predict_func_name = 'predict'
        
        self.model = model
        self.num_class: Optional[int] = None
        
        if isinstance(self.eval_metric, str):
            self.eval_metric = get_scorer(self.eval_metric)
        
        self._not_inner_model_params += ['eval_metric', 'num_class', 'device_type']

    def _prepare(
        self, 
        X: FeaturesType, 
        y: Optional[TargetType] = None, 
        categorical_feature: Optional[List[str]] = None
        ):
        X, y = self._prepare_data(X, y, categorical_feature)
        if y is not None:
            if self.model_type == "classification":
                self.num_class = np.unique(y).shape[0]
            return X, y
        else:
            return X
    
    def fit_fold(
        self, 
        X_train: FeaturesType, y_train: TargetType,
        X_test: FeaturesType, y_test: TargetType,
        inner_params: Dict[Any, Any] = {}
        ):
        fold_model = self.model(**inner_params)
        fold_model = fold_model.fit(X_train, y_train)
        fold_preds = getattr(fold_model, self.model_predict_func_name)(X_test)
        if self.model_type == "classification" and self.num_class == 2:
            score = self.eval_metric._score_func(y_test, fold_preds[:, 1])
        else:
            score = self.eval_metric._score_func(y_test, fold_preds)
        
        return fold_model, score, fold_preds
    
    def fit(self, X: FeaturesType, y: TargetType, categorical_feature: Optional[List[str]] = None):
        log.info(f"Fitting {self.name}", msg_type="start")

        X, y = self._prepare(X, y, categorical_feature)
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.random_state)
        cv = kf.split(X, y)

        self.models = []
        oof_preds = get_epmty_array(y.shape[0], self.num_class)
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            fold_model, _, fold_preds = self.fit_fold(
                X_train, y_train, 
                X_test, y_test,
                inner_params=self.inner_params)
            if self.model_type == "classification" and fold_preds.ndim == 1:
                fold_preds = np.vstack((1 - fold_preds, fold_preds)).T
            oof_preds[test_idx] = fold_preds
            self.models.append(fold_model)
        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    def optuna_objective(self, trial, X: FeaturesType, y: TargetType):
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.random_state)
        cv = kf.split(X, y)
        inner_params = {**self.not_tuned_params, **self.get_trial_params(trial),}
        
        # В sklearn 0.24.2 нет возможности передать scorer в cross_validate
        # scores = cross_validate(
        #     model,
        #     X, y,
        #     scoring=self.eval_metric,
        #     cv=cv,)
        scores = []
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            _, score, _ = self.fit_fold(
                X_train, y_train,
                X_test, y_test,
                inner_params=inner_params)
            scores.append(score)
        return np.mean(scores)

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        timeout: int=60,
        categorical_feature: Optional[List[str]] = None,
    ):
        log.info(f"Tuning {self.name}", msg_type="start")
            
        X, y = self._prepare(X, y, categorical_feature)
        study = tune_optuna(
            self.name,
            self.optuna_objective,
            X=X, y=y,
            greater_is_better=True,
            timeout=timeout,
            random_state=self.random_state,
        )
        self.best_params = study.best_params

        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")
        
    def _predict(self, X):
        """Predict on one dataset. Average all fold models"""
        X = self._prepare(X, categorical_feature=self.categorical_feature)
        y_pred = np.zeros((X.shape[0], self.num_class)) if self.num_class \
            else np.zeros((X.shape[0],))

        for fold_model in self.models:
            y_pred += getattr(fold_model, self.model_predict_func_name)(X)

        if self.model_type == "classification" and y_pred.ndim == 1:
            y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    @property
    def not_tuned_params(self):
        return {
            "random_state": self.random_state,
        }

    @property
    def inner_params(self) -> dict:
        return {
            **self.not_tuned_params,
            **{key: value for key, value in self.__dict__.items() if key not in self._not_inner_model_params},
            **self.best_params,
        }


class SLForestBase(SKBase):
    def __init__(
        self,
        model_type: str,
        model: BaseEstimator,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, ScorerType]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            model=model,
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            )
        self.verbose = 0 if isinstance(self.verbose, int) and self.verbose < 0 else self.verbose
        self.n_estimators: int = kwargs.pop('n_estimators', kwargs.pop('num_iterations', 1_000))
    
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @staticmethod
    def get_base_trial_params(trial):
        param_distr = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_float("min_samples_split", 0, 0.2),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0, 0.2),
            "max_features": trial.suggest_float("max_features", 0.1, 1),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }
        if param_distr["bootstrap"]:
            param_distr['max_samples'] = trial.suggest_float("max_samples", 0.5, 1)
            param_distr['oob_score'] = trial.suggest_categorical("oob_score", [True, False])
        return param_distr
        

class SLLinearBase(SKBase):
    def __init__(
        self,
        model_type: str,
        model: BaseEstimator,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, ScorerType]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            model=model,
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            )
        self._not_inner_model_params += ['num_iterations',]
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def get_base_trial_params(trial):
        param_distr = {
            'max_iter': trial.suggest_int("max_iter", 100, 15_000),
            'fit_intercept': trial.suggest_categorical("fit_intercept", [True, False,]),
        }
        return param_distr
