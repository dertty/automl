import pandas as pd
import numpy as np
from typing import TypeVar, Optional, List
from ..utils.utils import ArrayType
from ..utils.utils import get_array_type, check_array_type
from ..loggers import get_logger

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)

log = get_logger(__name__)

class NanFeatureSelector:
    '''
    Класс для отбора признаков с долей пропусков больше заданного значения.

    Attributes:
        nan_share_ts (float): Пороговое значение доли пропущенных значений.
    '''

    def __init__(self, nan_share_ts: float = 0.2) -> None:
        self.nan_share_ts = nan_share_ts

    def __call__(self, df: ArrayType) -> List[str]:
        # Directly ensure input is a DataFrame; assume these functions handle validation
        array_type = get_array_type(df)  
        check_array_type(array_type)

        # Calculate the share of NaNs directly without an unnecessary copy
        nan_share = df.isna().mean()
        # Select features where the share of NaNs meets or exceeds the threshold
        nan_features = nan_share[nan_share >= self.nan_share_ts].index.tolist()
        
        if len(nan_features) > 0:
            log.info(f"NaN features to drop: {nan_features}", msg_type="preprocessing")

        return nan_features

class QConstantFeatureSelector:
    '''
    Класс для отбора константных и квазиконстантных признаков.

    Attributes:
        feature_val_share_ts (float): Пороговое значение максимальной доли значения среди прочих значений признака.
    '''

    def __init__(self, feature_val_share_ts: float = 0.98) -> None:
        self.feature_val_share_ts = feature_val_share_ts

    def find_share_of_value(self, arr: pd.Series, col_name: str) -> Optional[str]:
        # Calculate the proportion of the most frequent value in the column
        arr_value_counts = arr.value_counts(normalize=True)
        max_arr_share = arr_value_counts.max()

        # Check if the proportion exceeds the threshold
        if max_arr_share >= self.feature_val_share_ts:
            return col_name
        return None
        
    def __call__(self, df: ArrayType) -> List[str]:
        # Validate the input type
        array_type = get_array_type(df)  
        check_array_type(array_type)

        # List comprehension to gather quasi-constant columns
        qconst_cols = [
            self.find_share_of_value(df[col], col) 
            for col in df.columns
        ]
        qconst_cols = [col for col in qconst_cols if col is not None]
        
        if len(qconst_cols) > 0:
            log.info(f"QConstant features to drop: {qconst_cols}", msg_type="preprocessing")
        # Filter out None values (columns not deemed quasi-constant)
        return qconst_cols

class ObjectColumnsSelector:
    '''
    Класс для отбора категориальных признаков с выбором стратегии кодирования в числовые признаки.

    Attributes:
        ohe_limiter (int): Максимальное число уникальных категорий для выбора стратегии OneHotEncoding.
        mode (str): Стратегия кодирования признаков.
    '''
    
    def __init__(self, ohe_limiter: int = 5, mode: str = 'ohe') -> None:
        if mode not in {'ohe', 'mte', 'oe'}:
            raise ValueError("Mode must be either 'ohe' or 'mte' or 'oe'.")
        
        self.ohe_limiter = ohe_limiter
        self.mode = mode

    def __call__(self, df: ArrayType) -> List[str]:
        # Ensure the input is correctly validated
        array_type = get_array_type(df)  
        check_array_type(array_type)

        # Handle only object (categorical) type columns
        df_obj = df.select_dtypes(include=["object", "category"])
        final_cols = df_obj.columns.tolist()
        # unique_counts = df_obj.nunique()
        
        # Depending on the mode, select columns accordingly
        # if self.mode == 'ohe':
        #     final_cols = unique_counts.index[unique_counts <= self.ohe_limiter].tolist()
        # else:
        #     final_cols = unique_counts.index[unique_counts > self.ohe_limiter].tolist()

        return final_cols

class CorrFeatureSelector:
    '''
    Класс для выявления зависимых признаков c помощью коэффициента корреляции Пирсона, коэффициента корреляции Спирмена.

    Attributes:
        corr_ts (float): Пороговое значение коэффициента корреляции двух переменных.
    '''
    
    def __init__(self, corr_ts: float = 0.8, corr_coef_method: str = 'pearson') -> None:
        self.corr_ts = corr_ts
        self.corr_coef_method = corr_coef_method

    def __call__(self, df: pd.DataFrame) -> List[str]:
        # Validate input to ensure it's a DataFrame
        array_type = get_array_type(df)  
        check_array_type(array_type)

        # Select only numeric columns directly, dropping NaN rows before computing correlation
        df_numeric = df.select_dtypes(include='number').dropna()

        # Compute Pearson correlation matrix
        corr_matrix = df_numeric.corr(method=self.corr_coef_method).abs()

        # Use upper triangle matrix to identify highly correlated columns
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)

        # List columns with correlations above the threshold
        corr_cols = [col for col in upper_corr.columns if any(upper_corr[col] > self.corr_ts)]

        return corr_cols
    
    
class SmartCorrelatedSelectionFast(BaseSelector):

    """
    Much faster version of `feature_engine.selection.SmartCorrelatedSelection`.
    Uses faster version of `find_correlated_features` function.
    """

    def __init__(
        self,
        variables=None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        selection_method: str = "missing_values",
        estimator=None,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        confirm_variables: bool = False,
    ):
        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError(
                f"`threshold` must be a float between 0 and 1. Got {threshold} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if selection_method not in [
            "missing_values",
            "cardinality",
            "variance",
            "model_performance",
        ]:
            raise ValueError(
                "selection_method takes only values 'missing_values', 'cardinality', "
                f"'variance' or 'model_performance'. Got {selection_method} instead."
            )

        if selection_method == "model_performance" and estimator is None:
            raise ValueError(
                "Please provide an estimator, e.g., "
                "RandomForestClassifier or select another "
                "selection_method."
            )

        if selection_method == "missing_values" and missing_values == "raise":
            raise ValueError(
                "When `selection_method = 'missing_values'`, you need to set "
                f"`missing_values` to `'ignore'`. Got {missing_values} instead."
            )

        super().__init__(confirm_variables)

        self.variables = variables
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values
        self.selection_method = selection_method
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        # check input dataframe
        X = check_X(X)

        self.variables_ = X.select_dtypes(include="number").columns.tolist()

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        if self.selection_method == "model_performance" and y is None:
            raise ValueError(
                "When `selection_method = 'model_performance'` y is needed to "
                "fit the transformer."
            )

        if self.selection_method == "missing_values":
            features = (
                X[self.variables_]
                .isnull()
                .sum()
                .sort_values(ascending=True)
                .index.to_list()
            )
        elif self.selection_method == "variance":
            features = (
                X[self.variables_].std().sort_values(ascending=False).index.to_list()
            )
        elif self.selection_method == "cardinality":
            features = (
                X[self.variables_]
                .nunique()
                .sort_values(ascending=False)
                .index.to_list()
            )
        else:
            features = sorted(self.variables_)

        correlated_groups, features_to_drop, correlated_dict = find_correlated_features(
            X, features, self.method, self.threshold
        )

        # select best performing feature according to estimator
        if self.selection_method == "model_performance":
            correlated_dict = dict()
            cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv
            for feature_group in correlated_groups:
                feature_performance, _ = single_feature_performance(
                    X=X,
                    y=y,
                    variables=feature_group,
                    estimator=self.estimator,
                    cv=cv,
                    groups=self.groups,
                    scoring=self.scoring,
                )
                # get most important feature
                f_i = (
                    pd.Series(feature_performance).sort_values(ascending=False).index[0]
                )
                correlated_dict[f_i] = feature_group.difference({f_i})

            # convoluted way to pick up the variables from the sets in the
            # order shown in the dictionary. Helps make transformer deterministic
            features_to_drop = [
                variable
                for set_ in correlated_dict.values()
                for variable in sorted(set_)
            ]

        self.features_to_drop_ = features_to_drop
        self.correlated_feature_sets_ = correlated_groups
        self.correlated_feature_dict_ = correlated_dict

        # save input features
        self._get_feature_names_in(X)

        return self