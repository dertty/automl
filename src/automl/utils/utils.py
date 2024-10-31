import pandas as pd
import numpy as np
import polars as pl

from typing import TypeVar, Optional, List


ArrayType = TypeVar('ArrayType', pd.DataFrame, pl.DataFrame, np.ndarray)


def get_array_type(array: ArrayType) -> str:
    if isinstance(array, pd.DataFrame):
        return 'pandas'
    elif isinstance(array, np.ndarray):
        return 'numpy'
    elif isinstance(array, pl.DataFrame):
        return 'polars'
    else:
        return ''


def check_array_type(type: str) -> None:
    match type:
        case 'pandas':
            pass
        case 'numpy':
            raise ValueError(f"make_column_selector can only be applied to pandas dataframes but not {type}")
        case 'polars':
            raise ValueError(f"make_column_selector can only be applied to pandas dataframes but not {type}")
        case _:
            raise ValueError(f"make_column_selector can only be applied to pandas dataframes but not {type}")