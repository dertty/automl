import numpy as np
import torch
import inspect
from catboost import CatBoostClassifier as CBClass
from catboost import CatBoostRegressor as CBReg
from catboost import Pool
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from typing import Any
from automl.metrics import ScorerWrapper
from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import convert_to_numpy, convert_to_pandas, tune_optuna
from .metrics import get_eval_metric

log = get_logger(__name__)


class CatBoostBase(BaseModel):
    __allowed = (x for x in \
        list(inspect.signature(CBClass.__init__).parameters.keys()) + \
            list(inspect.signature(CBReg.__init__).parameters.keys())\
                if x not in ['self'])
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
        time_series: bool = False,
        verbose: bool = False,
        task_type: str | None = None,
        n_jobs: int = -1,
        n_splits: int = 5,
        **kwargs,
    ):
        self.model_type = model_type
        match self.model_type:
            case 'classification':
                self.name = "CatBoostClassification"
                self.model = CBClass
                self.model_predict_func_name = 'predict_proba'
            case 'regression':
                self.name = "CatBoostRegression"
                self.model = CBReg
                self.model_predict_func_name = 'predict'
            case _:
                raise ValueError("Invalid model_type. Use 'classification' or 'regression'.")

        # model params
        self.thread_count = n_jobs or kwargs.pop("thread_count", 6)
        self.random_state = random_state
        self.verbose = verbose
        self.task_type = task_type or ("GPU" if torch.cuda.is_available() else "CPU")
        # other model params
        self.iterations = kwargs.pop('iterations', 2_000)
        self.max_iterations = self.iterations
        self.od_type = kwargs.pop('od_type', 'Iter')
        self.od_wait = kwargs.pop('od_wait', 100)
        self.od_pval = None if self.od_type == "Iter" else kwargs.pop('od_pval', 1e-5)
        
        # fit params
        self.time_series = time_series
        self.n_splits = n_splits
        self.cat_features: list[str] = []
        self.models: list[CBClass | CBReg] | None = None
        self.oof_preds = None
        self.eval_metric = None
        self.kf = (
            TimeSeriesSplit(n_splits=n_splits) if time_series else
            StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        )
        # tune params
        self.best_params: dict[str, Any] = {}
        
        for k, v in kwargs.items():
            assert(k in self.__class__.__allowed)
            setattr(self, k, v)
        

    def fit(self, X: FeaturesType, y: TargetType, categorical_features: list[str]=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.cat_features = categorical_features
        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(-1) if len(y.shape) == 1 or y.shape[1] == 1 else y
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
            fold_model = self.model(**self.inner_params)

            # fit/predict fold model
            fold_model.fit(train_data, eval_set=test_data)
            oof_preds[test_idx] = getattr(fold_model, self.model_predict_func_name)(test_data)

            # append fold model
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
            
    @staticmethod
    def get_trial_params(trial):
        raise NotImplementedError
        
    def objective(self, trial, X: FeaturesType, y: TargetType, scorer: ScorerWrapper):
        cv = self.kf.split(X, y)

        trial_params = self.get_trial_params(trial)
        not_tuned_params = self.not_tuned_params
        not_tuned_params['iterations'] = self.max_iterations

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

            fold_model = self.model(
                **trial_params, **not_tuned_params
            )

            fold_model.fit(train_data, eval_set=test_data)

            oof_preds[test_idx] = getattr(fold_model, self.model_predict_func_name)(test_data)
            best_num_iterations.append(fold_model.best_iteration_)
            models.append(fold_model)

        # add `iterations` as an optuna parameter
        trial.set_user_attr("iterations", round(np.mean(best_num_iterations)))

        # remove possible Nones in oof
        not_none_oof = np.where(np.logical_not(np.isnan(oof_preds[:, 0])))[0]

        if oof_preds.ndim == 2 and oof_preds.shape[1] == 2:
            # binary case
            trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof, 1])
        else:
            trial_metric = scorer.score(y[not_none_oof], oof_preds[not_none_oof])

        return trial_metric

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer: ScorerWrapper,
        timeout: int=60,
        categorical_features: list[str]=[],
    ):
        log.info(f"Tuning {self.name}", msg_type="start")

        self.cat_features = categorical_features
        self.eval_metric = get_eval_metric(scorer)

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(-1) if len(y.shape) == 1 or y.shape[1] == 1 else y
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
        # for key, val in study.best_params.items():
        #     setattr(self, key, val)
        self.iterations = study.best_trial.user_attrs["iterations"]
        self.best_params = study.best_params
        log.info(f"{len(study.trials)} trials completed", msg_type="optuna")
        log.info(f"{self.params}", msg_type="best_params")
        log.info(f"Tuning {self.name}", msg_type="end")

    def _predict(self, X_test: FeaturesType):
        """Predict on one dataset. Average all fold models"""
        X_test = convert_to_pandas(X_test)

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += getattr(fold_model, self.model_predict_func_name)(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def not_tuned_params(self) -> dict:
        not_tuned_params = {
            "thread_count": self.thread_count,
            "random_state": self.random_state,
            "verbose": self.verbose,
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
            **{key: value for key, value in self.__dict__.items() if key in self.__class__.__allowed},
            **self.best_params,
        }

    @property
    def meta_params(self) -> dict:
        return {
            "eval_metric": (
                self.eval_metric
                if isinstance(self.eval_metric, str) or self.eval_metric is None
                else "custom_metric"
            ),
            "time_series": self.time_series,
            'model_type': self.model_type,
            'model_predict_func_name': self.model_predict_func_name,
            'n_splits': self.n_splits,
        }

    @property
    def params(self) -> dict:
        return {
            **self.inner_params,
            **self.meta_params,
        }


class CatBoostClassification(CatBoostBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        verbose: bool = False,
        task_type: str | None = None,
        n_jobs: int = -1,
        n_splits: int = 5,
        **kwargs,
    ):
        super().__init__(
            model_type="classification",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            task_type=task_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
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
        task_type: str | None = None,
        n_jobs: int = -1,
        n_splits: int = 5,
        **kwargs,
    ):
        super().__init__(
            model_type="regression",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            task_type=task_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            **kwargs,
        )
    
    @staticmethod
    def get_trial_params(trial):
        # Получить базовые параметры из родительского класса
        params = CatBoostBase.get_base_trial_params(trial)

        # Добавить новые параметры для тюнинга
        params.update({
            "loss_function": trial.suggest_categorical(
                "loss_function", ["RMSE", "MAE", "MAPE",]
            ),
        })

        return params
