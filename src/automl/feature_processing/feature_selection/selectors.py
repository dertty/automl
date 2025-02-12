import numbers
import pandas as pd
import polars as pl
import numpy as np
from feature_engine.selection import SmartCorrelatedSelection, DropHighPSIFeatures
from feature_engine.outliers import Winsorizer
from sklearn.ensemble._bagging import _generate_indices
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils import Bunch, _safe_indexing, check_array, check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier, Pool
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from multiprocessing import cpu_count
from .CustomMetrics import regression_roc_auc_score
from ..loggers import get_logger
from .selectors import SmartCorrelatedSelectionFast

from automl.utils.model_utils import get_splitter, get_epmty_array
from .feature_selection.base_model_selector import BaseModelSelector
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType
from catboost import Pool
from typing import Optional, Any, List
from automl.model.type_hints import FeaturesType, TargetType


log = get_logger(__name__)

    
class CatboostShapFeatureSelectorCV(BaseModelSelector):
    def __init__(
        self,
        model_type: Optional[str] = None,
        random_state: int = 42,
        time_series: Optional[bool] = False,
        verbose: bool = False,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[str | Any] = None,
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
        #selection params
        self.n_features_to_select = kwargs.pop('n_features_to_select', None)
        self.complexity = kwargs.pop('complexity', None)
        self.algorithm = kwargs.pop('algorithm', None)
        self.steps = kwargs.pop('steps', 5)
        
        
        # model params
        self.thread_count = kwargs.pop('thread_count', self.n_jobs)
        self.task_type = kwargs.pop('task_type', self.device_type).upper()
        self.verbose = kwargs.pop('verbose', None) or kwargs.pop('verbose_eval', self.verbose)
        
        # other model params
        self.iterations: int = kwargs.pop('num_iterations', 2_000)
        self.iterations = kwargs.pop('iterations', None) or kwargs.pop('n_iterations', self.iterations)
        self.od_type = kwargs.pop('od_type', 'Iter')
        self.od_wait = kwargs.pop('early_stopping_rounds', None) or kwargs.pop('od_wait', 100)
        self.od_pval = None if self.od_type == "Iter" else kwargs.pop('od_pval', 1e-5)
        self.logging_level = kwargs.pop('logging_level', 'Silent')
        
        self.kwargs = kwargs
    
    @staticmethod
    def catboost_feature_selection(
        X_train, y_train, X_test, y_test,
        categorical_features: List[str] = [],
        n_features_to_select: Optional[int] =None, 
        logging_level: str = 'Silent',
        complexity: str = 'Approximate', 
        algorithm: str = 'RecursiveByPredictionValuesChange',
        steps: int = 5, 
        random_state: int = 42, 
        **kwargs
        ):
        assert complexity in ('Regular', 'Regular', 'Exact'), "Incorrect complexity. Choose 'Regular', 'Regular', 'Exact' complexity"
        assert algorithm in ('RecursiveByPredictionValuesChange', 'RecursiveByLossFunctionChange', 'RecursiveByShapValues'), "Incorrect algorithm. Choose 'RecursiveByPredictionValuesChange', 'RecursiveByLossFunctionChange', 'RecursiveByShapValues' algorithm"
        
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        test_pool = Pool(X_test, y_test, cat_features=categorical_features)
        
        model = CatBoostClassifier(random_state=random_state, logging_level=logging_level, **kwargs,)
        
        summary = model.select_features(
            train_pool, eval_set=test_pool,
            features_for_select=X_train.columns.to_list(),
            num_features_to_select=n_features_to_select if n_features_to_select else 1,
            train_final_model=False,
            logging_level=logging_level,
            algorithm=algorithm,
            shap_calc_type=complexity, 
            steps=steps)
        
        if n_features_to_select is None:
            optimal_idx = np.argmin(summary['loss_graph']['loss_values'])
            optimal_features = summary['selected_features_names'] + summary['eliminated_features_names'][::-1][:optimal_idx]
            return optimal_features
        else:
            return summary['selected_features_names']
    
    def transform(self, X: FeaturesType, y: Optional[TargetType] = None):
        if not hasattr(self, 'selected_features') or self.selected_features is None:
            raise RuntimeError("You must fit the transformer before calling transform.")
        return X[self.selected_features]

    def fit(self, X: FeaturesType, y: TargetType):
        log.info(f'Started feature selection.', msg_type="feature_selection")

        kf = get_splitter(
            self.model_type, 
            n_splits=self.n_splits, 
            time_series=self.time_series, 
            random_state=self.random_state)
        cv = kf.split(X, y)
        
        selected_features = set()
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train, y_train, X_test, y_test = self._get_train_test_data(X, y, train_idx, test_idx)
            optimal_features = self.catboost_feature_selection(
                X_train, y_train, X_test, y_test, 
                categorical_features = self.categorical_features or [],
                n_features_to_select = self.n_features_to_select, 
                logging_level = self.logging_level,
                complexity = self.complexity,
                algorithm = self.algorithm,
                steps = self.steps, 
                random_state = self.random_state, 
                allow_writing_files=False)
            
            selected_features |= set(optimal_features)
        
        self.selected_features = list(selected_features)
        log.info(f'Selected features: {self.selected_features}', msg_type="feature_selection")
        
        return self


class DropHighPSITransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        
        self.transformer.fit(X)
        self.psi_features_to_drop = self.transformer.features_to_drop_
        
        if len(self.psi_features_to_drop) > 0:
            log.info(f"Features not passing psi test to drop: {self.psi_features_to_drop}", msg_type="val_tests")
        
        return self
    
    def transform(self, X):
        X = X.drop(columns = self.psi_features_to_drop, axis=1)
        
        return X
    
class LowPSISelectoor(BaseModelSelector):
    def __init__(
        self,
        model_type: Optional[str] = None,
        random_state: int = 42,
        time_series: Optional[bool] = False,
        verbose: bool = False,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[str | Any] = None,
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
        self.threshold = kwargs.pop('threshold', 0.2)
        self.bins = kwargs.pop('bins', 15)
        self.strategy = kwargs.pop('strategy', 'equal_width')
        assert self.strategy in ('equal_width', 'equal_frequency'), "Incorrect strategy. Choose 'equal_width' or 'equal_frequency'"
        self.missing_values = kwargs.pop('missing_values', 'ignore')
        
        self.kwargs = kwargs
    
    def fit(self, X: FeaturesType, y: Optional[TargetType] = None):
        
        X = X.copy()
        kf = get_splitter(
            self.model_type, 
            n_splits=self.n_splits, 
            time_series=self.time_series, 
            random_state=self.random_state)
        cv = kf.split(X, y)
        
        for i, (train_idx, test_idx) in enumerate(cv):
            split_col = '__psi_split_col__'
            X[split_col] = 1
            X.iloc[train_idx, X.columns.get_loc(split_col)] = 0
            
            transformer = DropHighPSIFeatures(
                split_col=split_col,
                cut_off=0.5,
                threshold=self.threshold,
                bins=self.bins,
                strategy=self.strategy,
                missing_values=self.missing_values, 
                **self.kwargs)
            
            transformer = transformer.fit(X)
            selected_features = [col for col in X.columns if col not in self.transformer.features_to_drop_]
        
    def transform(self, X: FeaturesType, y: Optional[TargetType] = None):
        
        