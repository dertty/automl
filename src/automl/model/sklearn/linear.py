import warnings

from sklearn.linear_model import LogisticRegression as LogRegSklearn
from sklearn.linear_model import Ridge
from typing import Optional, Union
from ...loggers import get_logger
from ..type_hints import ScorerType
from .base import SLLinearBase


warnings.filterwarnings("ignore")


log = get_logger(__name__)


class LogisticRegression(SLLinearBase):
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
            model=LogRegSklearn,
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
        params = SLLinearBase.get_base_trial_params(trial)
        params['C'] =  trial.suggest_float("C", 1e-3, 1e3, log=True)
        params['solver'] = trial.suggest_categorical('solver', [
            # 'liblinear', 
            'saga', 
            # 'lbfgs',
            # 'newton-cg'
            ])
        
        if params['solver'] == 'saga':
            params['penalty'] = trial.suggest_categorical("penalty", ['none', 'l2', 'l1', 'elasticnet',])
        elif params['solver'] == 'liblinear':
            params['penalty'] = trial.suggest_categorical("penalty", ['l2', 'l1',])
        else:
            params['penalty'] = trial.suggest_categorical("penalty", ['l2', 'none',])
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        return params


class RidgeRegression(SLLinearBase):
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
            model=Ridge,
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
        self._not_inner_model_params += ['n_jobs', 'verbose',]
    
    @staticmethod
    def get_trial_params(trial):
        params = SLLinearBase.get_base_trial_params(trial)
        params['alpha'] = trial.suggest_float("alpha", 1e-6, 1e6, log=True)
        return params