import os
import enum
import pandas as pd
import torch

from typing import Any, Optional, List, Callable, Dict, Union
from .type_hints import FeaturesType, TargetType
from .utils import convert_to_numpy, convert_to_pandas


class BaseModel:
    """
    General structure of a model.

    All models have common `predict` method.

    However, the method `_predict` that actually predicts the underlying model
        should be implemented for each model separately
    """

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
    ):
        self.model_type = model_type
        if self.model_type not in ('classification', 'regression'):
            raise ValueError("Invalid model_type. Use 'classification' or 'regression'.")
        self.n_jobs = n_jobs or max(1, int(os.cpu_count() / 2))
        try:
            is_cuda_available = torch.cuda.is_available()
            self.device_type = device_type or ("CUDA" if is_cuda_available else "CPU")
        except (ImportError, AttributeError):
            self.device_type = device_type or "CPU"
        self.random_state = random_state
        self.verbose = verbose
        self.time_series = time_series
        self.eval_metric = eval_metric
        self.n_splits = n_splits
        
        self.oof_preds = None
        self.categorical_feature: Optional[List[str]] = None
        self.best_params: Dict[str, Any] = {}
        
        self._not_inner_model_params = [
            'name', 'num_class', 'max_iterations', 
            'categorical_feature', 'models', 'oof_preds', 
            'best_params', 'time_series', 'n_splits',
            '_not_inner_model_params', 'model_type', 'model',
            'model_predict_func_name'
            ]
    
    def _prepare_categorical_features(self, X: FeaturesType, categorical_feature: Optional[List[Union[str, int]]] = None):
        categorical_feature = categorical_feature or []
        if not isinstance(X, pd.DataFrame):
            self.categorical_feature = [f"column_{i}" for i in categorical_feature if i < len(X)]
        else:
            self.categorical_feature = [col for col in categorical_feature if col in X.columns]
             
    def _prepare_data(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_feature: Optional[List[Union[str, int]]] = None):
        self._prepare_categorical_features(X, categorical_feature)
        X = convert_to_pandas(X)
        if y is not None:
            # y = convert_to_numpy(y)
            if y.ndim == 1:
                y = y.reshape(-1)
            elif y.ndim == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if self.model_type == "regression" and y.ndim > 1:
                raise ValueError("y must be 1D for regression")
        return X, y
    
    def _prepare(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_feature: Optional[List[Union[str, int]]] = None):
        raise NotImplementedError
        
    def fit(self, X: FeaturesType, y: TargetType, categorical_feature: Optional[List[str]] = None):
        """
        Fit the model to the provided data.

        Parameters:
        X (FeaturesType): The input features.
        y (TargetType): The target values.
        categorical_feature (List[str]): List of categorical features.

        Raises:
        NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def tune(self, X: FeaturesType, y: TargetType, timeout: int, categorical_feature: Optional[List[str]] = None):
        """
        Tune the model's hyperparameters.

        Parameters:
        X (FeaturesType): The input features.
        y (TargetType): The target values.
        timeout (int): The maximum time allowed for tuning.
        categorical_feature (List[str]): List of categorical features.

        Raises:
        NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    @staticmethod
    def get_base_trial_params(trial):
        raise NotImplementedError
    
    @staticmethod
    def get_trial_params(trial):
        raise NotImplementedError

    def _predict(self, X: FeaturesType):
        raise NotImplementedError
    
    @property
    def not_tuned_params(self):
        raise NotImplementedError

    @property
    def inner_params(self):
        raise NotImplementedError

    @property
    def meta_params(self) -> dict:
        return {
            "time_series": self.time_series,
            'model_type': self.model_type,
            'n_splits': self.n_splits,
            'name': self.name,
            'eval_metric': self.eval_metric
                if isinstance(self.eval_metric, str) or self.eval_metric is None
                else "custom_metric"
        }

    @property
    def params(self) -> dict:
        return {
            **self.inner_params,
            **self.meta_params,
        }
    
    def predict(self, Xs):
        """
        Predict the target values for the given input features.
        
        Parameters:
        Xs (FeaturesType or list of FeaturesType): The input features.
        
        Returns:
        TargetType or list of TargetType: The predicted target values.
        
        Raises:
        ValueError: If the model has not been fitted yet.
        """
        ### BUG
        # if isinstance(Xs, FeaturesType):
        # TypeError: Subscripted generics cannot be used with class and instance checks in < python3.10
        if not hasattr(self, 'models') and not hasattr(self, 'model'):
            raise ValueError("The model has not been fitted yet. Please call the `fit` method before `predict`.")
        if hasattr(self, 'models') and (self.models is None or self.models == []):
            raise ValueError("The model has not been fitted yet. Please call the `fit` method before `predict`.")
        if hasattr(self, 'model') and self.model is None:
            raise ValueError("The model has not been fitted yet. Please call the `fit` method before `predict`.")
            
        # If Xs is not a list or the model is a Blender, perform a single prediction
        if not isinstance(Xs, list) or self.name == "Blender":
            return self._predict(Xs)

        # several tests
        ys_pred = []
        for X in Xs:
            y_pred = self._predict(X)
            ys_pred.append(y_pred)

        return ys_pred
    
    @staticmethod
    def _get_train_test_data(X: FeaturesType, y: TargetType, train_idx, test_idx):
        """
        Split the data into training and testing sets based on provided indices.

        Parameters:
        X (FeaturesType): The input features.
        y (TargetType): The target values.
        train_idx (array-like): Indices for the training set.
        test_idx (array-like): Indices for the testing set.

        Returns:
        tuple: A tuple containing the training features, training targets, testing features, and testing targets.
        """
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def get_str_greater_is_better(model_type: str, eval_metric: str, metrics_dict: Dict[str, bool]):
        if eval_metric in metrics_dict.keys():
            greater_is_better = metrics_dict[eval_metric]
        elif model_type == 'classification':
            greater_is_better = True
        elif model_type == 'regression':
            greater_is_better = False
        else:
            greater_is_better = None
        return greater_is_better


@enum.unique
class ModelType(enum.Enum):
    BASE_MODEL = 0
    BLENDER = 1
    STACKER = 2
