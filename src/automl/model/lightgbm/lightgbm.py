import lightgbm as lgb
import numpy as np

from typing import Any, Optional, List, Dict, Callable, Union
from ...loggers import get_logger
from ..base import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import tune_optuna
from ..utils.model_utils import get_splitter, get_epmty_array


log = get_logger(__name__)


class LightGBMBase(BaseModel):
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
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
            self.name: str = "LightGBMClassification"
            self.objective: str = "binary" # or multiclass, multilabel not allowed
        elif self.model_type == 'regression':
            self.name = "LightGBMRegression"
            self.objective = "regression"

        # Так как в модель передаются все атрибуты класса,
        # то все атрибуты класса должны соответствовать параметрам модели
        # лишнее либо удаляем, либо добавляем в исключения
        # model params
        self.num_threads: int = kwargs.pop('num_threads', self.n_jobs)
        self.seed: int = kwargs.pop('seed', self.random_state)
        self.verbosity: int = kwargs.pop('verbosity', self.verbose)
        self.device_type: str = self.device_type.lower()
        # other model params
        num_iterations_aliases = [
            'iterations', 'num_iteration', 'n_iter', 'num_tree', 
            'num_trees', 'num_round', 'num_rounds', 'nrounds', 
            'num_boost_round', 'n_estimators', 'max_iter']
        num_iterations, kwargs = self._get_param_value(kwargs, num_iterations_aliases, 2_000)
        self.num_iterations: Optional[int] = num_iterations
        self.max_iterations: Optional[int] = num_iterations
        early_stopping_round_aliases = [
            'early_stopping_round', 'early_stopping_rounds', 
            'early_stopping', 'n_iter_no_change',]
        early_stopping_round, kwargs = self._get_param_value(kwargs, early_stopping_round_aliases, 100)
        self.early_stopping_round: Optional[int] = early_stopping_round
        self.early_stopping_min_delta = kwargs.pop('early_stopping_min_delta', 1e-4)
        if ('scale_pos_weight' in kwargs or 'class_weight' in kwargs) and 'is_unbalance' in kwargs:
            raise ValueError("You can't use both `class_weight` or `scale_pos_weight` and `is_unbalance`")
        elif 'scale_pos_weight' in kwargs or 'class_weight' in kwargs:
            self.scale_pos_weight: float = kwargs.pop('class_weight', None) or kwargs.pop('scale_pos_weight', 1.0)
        elif 'is_unbalance' in kwargs:
            self.is_unbalance: str = kwargs.pop('is_unbalance')
        # fit params
         
        self.num_class: Optional[int] = None
        if isinstance(self.eval_metric, str):
            self.metric: Optional[str] = self.eval_metric
        elif callable(self.eval_metric):
            self.metric = 'None'
        else:
            self.metric = ''
        
        self._not_inner_model_params += ['n_jobs', 'random_state', 'verbose', 'eval_metric']
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _get_param_value(kwargs: Dict[str, Any], keys: List[str], default_value: Any):
        # Перебираем все возможные варианты и удаляем из kwargs
        result = None
        for key in keys:
            value = kwargs.pop(key, None)
            if value:
                result = result or value
        result = result or default_value
        return result, kwargs
        
    def _prepare(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_feature: Optional[List[str]] = None):
        X, y = self._prepare_data(X, y, categorical_feature)
        if y is not None:
            if self.model_type == "classification":
                self.num_class = np.unique(y).shape[0]
                # correct objective based on the number of classes
                if self.objective == 'binary' and self.num_class > 2:
                    self.objective = "multiclass"
            return X, y
        else:
            return X

    @staticmethod
    def _fix_params_name(params: dict):
        """
        Fix parameter names to avoid LightGBM alias warnings.

        LightGBM produces warnings when using certain parameter aliases.
        This method renames the input parameters to their canonical names
        to avoid these warnings. In recent versions of LightGBM, this issue
        has been resolved, but this method ensures compatibility with older versions.

        Args:
            params (dict): Dictionary of parameters to be fixed.

        Returns:
            dict: Dictionary with fixed parameter names.
        """
        params_upd = {**params}
        params_upd["colsample_bytree"] = params_upd.get("colsample_bytree") or params_upd.pop("feature_fraction", None)
        params_upd["reg_alpha"] = params_upd.get("reg_alpha") or params_upd.pop("lambda_l1", None)
        params_upd["subsample"] = params_upd.get("subsample") or params_upd.pop("bagging_fraction", None)
        params_upd["min_child_samples"] = params_upd.get("min_child_samples") or params_upd.pop("min_data_in_leaf", None)
        params_upd["min_split_gain"] = params_upd.get("min_split_gain") or params_upd.pop("min_gain_to_split", None)
        params_upd["subsample_freq"] = params_upd.get("subsample_freq") or params_upd.pop("bagging_freq", None)
        params_upd["reg_lambda"] = params_upd.get("reg_lambda") or params_upd.pop("lambda_l2", None)
        params_upd["boosting_type"] = params_upd.pop("boosting", None)
        return {key:  value for key, value in params_upd.items() if value}

    def fit_fold(
        self, 
        X_train: FeaturesType, y_train: TargetType,
        X_test: FeaturesType, y_test: TargetType,
        inner_params: Dict[Any, Any] = {}
        ):
        train_data = lgb.Dataset(
            X_train,
            y_train,
            categorical_feature=self.categorical_feature,
        )
        test_data = lgb.Dataset(
            X_test,
            y_test,
            categorical_feature=self.categorical_feature,
        )
        feval = None
        if self.metric == 'None':
            feval = self.eval_metric
        # fit/predict fold model
        fold_model = lgb.train(
            params=inner_params,
            train_set=train_data,
            valid_sets=[test_data],
            valid_names=['test_data'], 
            feval = feval,
            # num_boost_round=self.num_iterations,
            callbacks=[
                lgb.early_stopping(
                    self.early_stopping_round or -1, 
                    first_metric_only=False, 
                    verbose=self.verbosity > 0, 
                    min_delta=self.early_stopping_min_delta
                    )
                ],
            )
        best_score_dict: Dict[str, float] = fold_model.best_score['test_data']
        for _, value in best_score_dict.items():
            best_score: float = value
        return fold_model, best_score
    
    def fit(self, X: FeaturesType, y: TargetType, categorical_feature: Optional[List[str]] = None):
        log.info(f"Fitting {self.name}", msg_type="start")

        X, y = self._prepare(X, y, categorical_feature or [])
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.seed)
        cv = kf.split(X, y)
        
        self.models = []
        oof_preds = get_epmty_array(y.shape[0], self.num_class)
        for i, (train_idx, test_idx) in enumerate(cv):
            # log.info(f"{self.name} fold {i}", msg_type="fit")

            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            fold_model, _ = self.fit_fold(
                X_train, y_train, 
                X_test, y_test,
                inner_params=self._fix_params_name(self.inner_params))
            
            preds = fold_model.predict(X_test)
            if self.model_type == "classification" and preds.ndim == 1:
                preds = np.vstack((1 - preds, preds)).T
            oof_preds[test_idx] = preds
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    @staticmethod
    def get_base_trial_params(trial):
        default_param_distr = {
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "num_leaves": trial.suggest_int("num_leaves", 10, 512),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 256),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 20, step=5),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 20),
        }

        return default_param_distr
    
    def optuna_objective(self, trial, X: FeaturesType, y: TargetType):
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.seed)
        cv = kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params
        not_tuned_params['num_iterations'] = self.max_iterations
        inner_params = self._fix_params_name({**not_tuned_params, **trial_params})
        
        # oof_preds = get_epmty_array(y.shape[0], None if self.num_class == 2 else self.num_class)
        best_num_iterations = []
        scores = []
        for i, (train_idx, test_idx) in enumerate(cv):
            # log.info(f"{self.name} fold {i}", msg_type="fit")
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            fold_model, score = self.fit_fold(
                X_train, y_train,
                X_test, y_test,
                inner_params=inner_params)
            scores.append(score)
            # oof_preds[test_idx] = fold_model.predict(X_test)
            best_num_iterations.append(fold_model.current_iteration())
        # add `num_iterations` to the optuna parameters
        trial.set_user_attr("num_iterations", round(np.mean(best_num_iterations)))
        return np.mean(scores)
        # remove possible Nones in oof
        # if oof_preds.ndim == 1:
        #     not_none_oof = ~np.isnan(oof_preds).any(axis=1)
        # else:
        #     not_none_oof = ~np.isnan(oof_preds)

        # if oof_preds.ndim == 2 and oof_preds.shape[1] == 2:
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
            _, _, greater_is_better = self.eval_metric(
                [1, 0, 1], 
                lgb.Dataset(
                    data=np.expand_dims([1, 0, 1], axis=1), 
                    label=[1, 0, 1]
                    ).construct())
        else:
            greater_is_better = None
            
        X, y = self._prepare(X, y, categorical_feature)
        study = tune_optuna(
            self.name,
            self.optuna_objective,
            X=X, y=y,
            greater_is_better=greater_is_better,
            timeout=timeout,
            random_state=self.seed,)
        self.best_params = study.best_params
        self.best_params['num_iterations'] = study.best_trial.user_attrs["num_iterations"]
        
        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X):
        """Predict on one dataset. Average all fold models"""
        X = self._prepare(X, categorical_feature=self.categorical_feature)
        y_pred = np.zeros((X.shape[0], self.num_class)) if self.num_class and self.num_class > 2 \
            else np.zeros((X.shape[0],))

        for fold_model in self.models:
            y_pred += fold_model.predict(X)
        y_pred = y_pred / len(self.models)
        
        if self.model_type == "classification" and y_pred.ndim == 1:
            y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    @property
    def not_tuned_params(self) -> dict:
        not_tuned_params = {
            "num_threads": self.num_threads,
            "seed": self.seed,
            "verbosity": self.verbosity,
            "device_type": self.device_type,
            "early_stopping_round": self.early_stopping_round,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            'objective': self.objective,
            'metric': self.metric,
        }
        return {key: value for key, value in not_tuned_params.items() if value is not None}

    @property
    def inner_params(self) -> dict:
        return {
            'num_iterations': self.num_iterations,
            **self.not_tuned_params,
            **{key: value for key, value in self.__dict__.items() if key not in self._not_inner_model_params},
            **self.best_params,
        }


class LightGBMClassification(LightGBMBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
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
        params = LightGBMBase.get_base_trial_params(trial)
        params["is_unbalance"] = trial.suggest_categorical("is_unbalance", ['true', 'false'])
        # if params["is_unbalance"] == 'false':
        #     params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0, 1)
        return params


class LightGBMRegression(LightGBMBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
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
        params = LightGBMBase.get_base_trial_params(trial)
        params.update({
            "objective": trial.suggest_categorical(
                "objective", ["regression", "regression_l1", "huber"]
            ),
        })

        return params