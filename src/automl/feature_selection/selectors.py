import pandas as pd
import numpy as np
from typing import TypeVar, Optional, List
from automl.utils.utils import ArrayType
from automl.utils.utils import get_array_type, check_array_type

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
        
        # Filter out None values (columns not deemed quasi-constant)
        return [col for col in qconst_cols if col is not None]

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
        df_obj = df.select_dtypes(include='object')
        unique_counts = df_obj.nunique()
        
        # Depending on the mode, select columns accordingly
        if self.mode == 'ohe':
            final_cols = unique_counts.index[unique_counts <= self.ohe_limiter].tolist()
        else:
            final_cols = unique_counts.index[unique_counts > self.ohe_limiter].tolist()

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