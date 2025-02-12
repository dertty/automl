import os
import enum
from typing import Any, Optional, List, Callable, Dict, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from automl.model.type_hints import FeaturesType, TargetType
from automl.model.utils import convert_to_numpy, convert_to_pandas


class BaseModelSelector(BaseEstimator, TransformerMixin):
    """
    General structure of a selector based on importance values.
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        random_state: int = 42,
        time_series: Optional[bool] = False,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
    ):
        self.model_type = model_type
        self.n_jobs = n_jobs or max(1, int(os.cpu_count() / 2))
        try:
            import torch
            is_cuda_available = torch.cuda.is_available()
            self.device_type = device_type or ("CUDA" if is_cuda_available else "CPU")
        except (ImportError, AttributeError):
            self.device_type = device_type or "CPU"
        self.random_state = random_state
        self.verbose = verbose
        self.time_series = time_series
        self.eval_metric = eval_metric
        self.n_splits = n_splits
        
        self.categorical_features: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        
    def _prepare_data(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_features: Optional[List[Union[str, int]]] = None):
        categorical_features = categorical_features or []
        if not isinstance(X, pd.DataFrame):
            self.categorical_features = [f"column_{i}" for i in categorical_features if i < len(X)]
        else:
            self.categorical_features = [col for col in categorical_features if col in X.columns]
        X = convert_to_pandas(X)
        if y is not None:
            y = convert_to_numpy(y)
            if y.ndim == 1:
                y = y.reshape(-1)
            elif y.ndim == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if self.model_type == "regression" and y.ndim > 1:
                raise ValueError("y must be 1D for regression")
        return X, y
    
    def _prepare(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_features: Optional[List[Union[str, int]]] = None):
        raise NotImplementedError
        
    def fit(self, X: FeaturesType, y: TargetType):
        raise NotImplementedError
    
    def transform(self, X: FeaturesType, y: Optional[TargetType] = None):
        raise NotImplementedError
    
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
