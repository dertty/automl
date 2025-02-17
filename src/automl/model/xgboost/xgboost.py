import numpy as np
from sklearn.utils import compute_sample_weight
import xgboost as xgb

from typing import Any, Optional, Callable, Union, List, Dict
from ...loggers import get_logger
from ..base import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import tune_optuna
from ..utils.model_utils import get_splitter, get_epmty_array


log = get_logger(__name__)


class XGBBase(BaseModel):
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: int = None,
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
            self.name: str = "XGBClassification"
            self.objective: str = "binary:logistic" # or reg:logistic, multilabel not allowed
            self.model_predict_func_name: str = 'predict_proba'
        elif self.model_type == 'regression':
            self.name = "XGBRegression"
            self.objective = "reg:squarederror"
            self.model_predict_func_name = 'predict'
        
        # model params
        self.nthread: int = kwargs.pop('nthread', self.n_jobs)
        self.seed: int = kwargs.pop('seed', self.random_state)
        self.verbosity: int = kwargs.pop('verbosity', self.verbose)
        self.verbosity = self.verbosity if self.verbosity >= 0 else 0
        self.device: str = kwargs.pop('device', self.device_type).lower()
        # other model params
        self.num_boost_round: int = kwargs.pop('num_iterations', 2_000)
        self.num_boost_round = kwargs.pop('num_boost_round', self.num_boost_round)
        self.max_iterations: int = self.num_boost_round
        self.early_stopping_rounds: int = kwargs.pop('early_stopping_rounds', 100)
        # fit params
        self.models: Optional[List[xgb.core.Booster]] = None
        self.class_weight = kwargs.pop('class_weight', 'balanced')
        self.num_class: Optional[int] = None
        if isinstance(self.eval_metric, str):
            self.eval_metric: Optional[str] = self.eval_metric.lower()
        elif callable(self.eval_metric):
            self.disable_default_eval_metric = 1
        
        self._not_inner_model_params += ['n_jobs', 'random_state', 'verbose', 'eval_metric', 'device_type',]
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def _prepare(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_feature: Optional[List[str]] = None):
        X, y = self._prepare_data(X, y, categorical_feature or [])
        X.loc[:, self.categorical_feature] = X[self.categorical_feature].astype("category")
        if y is not None:
            if self.model_type == "classification":
                self.num_class = np.unique(y).shape[0]
                # correct objective based on the number of classes
                if self.objective == 'binary:logistic' and self.num_class > 2:
                    self.objective = "multi:softmax"
            return X, y
        else:
            return X

    def fit_fold(
        self, 
        X_train: FeaturesType, y_train: TargetType,
        X_test: FeaturesType, y_test: TargetType,
        inner_params: Dict[Any, Any] = {},
        ):
        # add class weights
        # in binary case -> `scale_pos_weight`
        # in multiclass case -> `sample_weight`
        sample_weight = compute_sample_weight(
            class_weight=None, y=y_train)
        if self.model_type == "classification":
            if self.class_weight == "balanced":
                if self.num_class == 2:
                    # binary case
                    class_count = np.bincount(y_train.astype(int))
                    self.scale_pos_weight = class_count[0] / class_count[1]
                    inner_params["scale_pos_weight"] = self.scale_pos_weight
                elif self.num_class and self.num_class > 2:
                    # multiclass case
                    sample_weight = compute_sample_weight(
                        class_weight="balanced", y=y_train
                        )
        dtrain = xgb.DMatrix(
            X_train, label=y_train, 
            weight=sample_weight,
            silent=True, nthread=self.nthread, enable_categorical=True,)
        dtest = xgb.DMatrix(
            X_test, label=y_test, 
            silent=True, nthread=self.nthread, enable_categorical=True,)
        
        custom_metric = None
        if callable(self.eval_metric):
            custom_metric = self.eval_metric
        fold_model = xgb.train(
            params=inner_params, 
            dtrain=dtrain, 
            num_boost_round=self.num_boost_round,
            evals=[(dtest, 'test',),], 
            early_stopping_rounds=self.early_stopping_rounds, 
            evals_result=None, 
            verbose_eval=self.verbosity, 
            callbacks=None, 
            custom_metric=custom_metric,)
        return fold_model, fold_model.best_score
        
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
                inner_params=self.inner_params)

            dtest = xgb.DMatrix(
                X_test, 
                silent=True, 
                nthread=self.nthread, 
                enable_categorical=True,
                )
            preds = fold_model.predict(dtest)
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

        return default_param_distr

    def optuna_objective(self, trial, X: FeaturesType, y: TargetType):
        """
        Perform cross-validation to evaluate the model.
        Mean test score is returned.
        """
        kf = get_splitter(self.model_type, n_splits=self.n_splits, time_series=self.time_series, random_state=self.seed)
        cv = kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params
        not_tuned_params['num_boost_round'] = self.max_iterations
        inner_params = {**not_tuned_params, **trial_params, }
        
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

            # dtest = xgb.DMatrix(
            #     X_test, 
            #     silent=True, 
            #     nthread=self.nthread, 
            #     enable_categorical=True,)
            scores.append(score)
            # oof_preds[test_idx] = fold_model.predict(dtest)
            if self.early_stopping_rounds >= 0:
                best_num_iterations.append(fold_model.best_iteration)
            else:
                best_num_iterations.append(self.num_boost_round)
        # add `n_estimators` to the optuna parameters
        trial.set_user_attr("num_boost_round", round(np.mean(best_num_iterations)))
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
            _, _, greater_is_better = self.eval_metric()
        else:
            greater_is_better = None
        
        X, y = self._prepare(X, y, categorical_feature or [])
        study = tune_optuna(
            self.name,
            self.optuna_objective,
            X=X, y=y,
            greater_is_better=greater_is_better,
            timeout=timeout,
            random_state=self.seed,
        )
        self.best_params = study.best_params
        self.best_params['num_boost_round'] = study.best_trial.user_attrs["num_boost_round"]
        
        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X):
        """Predict on one dataset. Average all fold models"""
        X = self._prepare(X, categorical_feature=self.categorical_feature)
        y_pred = np.zeros((X.shape[0], self.num_class)) if self.num_class and self.num_class > 2 \
            else np.zeros((X.shape[0],))
        
        dtest = xgb.DMatrix(
                X, 
                silent=True, 
                nthread=self.nthread, 
                enable_categorical=True,
                )
        
        for fold_model in self.models:
            y_pred += fold_model.predict(dtest)
        y_pred = y_pred / len(self.models)
        
        if self.model_type == "classification" and y_pred.ndim == 1:
            y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    @property
    def not_tuned_params(self) -> dict:
        not_tuned_params = {
            "nthread": self.nthread,
            "seed": self.seed,
            "verbosity": self.verbosity,
            "device": self.device,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
        }
        return {key: value for key, value in not_tuned_params.items() if value}

    @property
    def inner_params(self) -> dict:
        return {
            **self.not_tuned_params,
            **{key: value for key, value in self.__dict__.items() if key not in self._not_inner_model_params},
            **self.best_params,
        }

 
class XGBClassification(XGBBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: int = None,
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
        params = XGBBase.get_base_trial_params(trial)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return params


class XGBRegression(XGBBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: int = 0,
        device_type: Optional[str] = None,
        n_jobs: int = None,
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
        params = XGBBase.get_base_trial_params(trial)
        # params.update({
        #     "objective": trial.suggest_categorical(
        #         "objective", ["regression", "regression_l1", "huber"]
        #     ),
        # })

        return params
