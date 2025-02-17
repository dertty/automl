from sklearn.ensemble import ExtraTreesClassifier as EXClasSklearn
from sklearn.ensemble import ExtraTreesRegressor as EXRegSklearn
from sklearn.ensemble import RandomForestClassifier as RFClassSklearn
from sklearn.ensemble import RandomForestRegressor as RFRegSklearn

from typing import Optional, Union

from ...loggers import get_logger
from ..type_hints import ScorerType
from .base import SLForestBase


log = get_logger(__name__)


class ExtraTreesClassification(SLForestBase):
    def __init__(
        self,
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
            model = EXClasSklearn,
            model_type="classification",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric or 'roc_auc',
            **kwargs,
        )
    
    @staticmethod   
    def get_trial_params(trial):
        params = SLForestBase.get_base_trial_params(trial)
        params["criterion"] = trial.suggest_categorical("criterion", [
            "gini", 
            "entropy", 
            # "log_loss", # log_loss is not supported by ExtraTreesClassifier in 0.24.2
            ])
        params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
        return params


class ExtraTreesRegression(SLForestBase):
    def __init__(
        self,
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
            model = EXRegSklearn,
            model_type="regression",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric or 'neg_mean_squared_error',
            **kwargs,
        )
    
    @staticmethod
    def get_trial_params(trial):
        params = SLForestBase.get_base_trial_params(trial)

        return params

   
class RandomForestClassification(SLForestBase):
    def __init__(
        self,
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
            model=RFClassSklearn,
            model_type="classification",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric or 'roc_auc',
            **kwargs,
        )
    
    @staticmethod   
    def get_trial_params(trial):
        params = SLForestBase.get_base_trial_params(trial)
        params["criterion"] = trial.suggest_categorical("criterion", [
            "gini", 
            "entropy", 
            # "log_loss", # log_loss is not supported by RandomForestClassifier in 0.24.2
            ])
        params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
        return params


class RandomForestRegression(SLForestBase):
    def __init__(
        self,
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
            model=RFRegSklearn,
            model_type="regression",
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric or 'neg_mean_squared_error',
            **kwargs,
        )
    
    @staticmethod
    def get_trial_params(trial):
        params = SLForestBase.get_base_trial_params(trial)

        return params