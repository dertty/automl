import optuna
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from typing import Tuple, Callable, Union, Type
from .type_hints import FeaturesType, TargetType
from ..utils import optuna_tune, convert_to_numpy, convert_to_pandas


class BaseModel:
    """
    General structure of a model.

    All models have common `predict` method.

    However, the method `_predct` that actually predicts the underlying model
        should be implemented for each model separately
    """

    def __init__(self):
        pass

    def fit(self, X: FeaturesType, y: TargetType):
        raise NotImplementedError

    def _predict(self, X_test):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    def predict(self, Xs):
        ### BUG
        # if isinstance(Xs, FeaturesType):
        # TypeError: Subscripted generics cannot be used with class and instance checks in < python3.10

        if not isinstance(Xs, list):
            # only one test
            return self._predict(Xs)

        # several tests
        ys_pred = []
        for X_test in Xs:
            y_pred = self._predict(X_test)
            ys_pred.append(y_pred)

        return ys_pred


class BaseTuner:
    def __init__(
        self, estimator: BaseModel, 
        X: FeaturesType, y: TargetType, 
        metric: Callable, timeout: int=60,
        mode: str='fast'
    ):
        self.model = estimator
        self.estimator = estimator.__class__
        self.model_params = {attr: getattr(self.model, attr) for attr in self.model.__dict__}
        self.hyperparams = self.model.params
        self.mode = mode
        self.timeout = timeout
        self.metric = metric
  
        self.X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        self.y = y.reshape(y.shape[0])
        self.X_shape = X.shape
    
    def get_trial_params(self, trial: optuna.Trial) -> dict:
        match self.mode:
            case 'fast':
                pass
            case 'balanced':
                pass
            case _:
                pass
        raise NotImplementedError
     
    def objective(
        self, 
        trial: optuna.Trial, 
        trial_params_func: Callable,
    ) -> float:
        trial_params = trial_params_func(trial)
        params = {**self.model_params, **self.hyperparams, **trial_params}
            
        model = self.estimator(**params)
        oof_preds = model.fit(self.X, self.y)
        
        score = self.metric(self.y, oof_preds)
        return score
    
    def tune(self):
        optuna_early_stopping_rounds = 20 if self.mode == 'fast' else 50 if self.mode == 'balanced' else 100
        study = optuna_tune(name=self.model_params['name'], objective=self.objective, trial_params_func=self.get_trial_params, early_stopping_rounds=optuna_early_stopping_rounds, random_state=self.model_params['random_state'])

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self.model, key, val)
            
        return self.model



        
            