import numpy as np
import pandas as pd

from .conversions import convert_to_numpy


def prepare_time_series(X, y, time_index_col):
    # checks for time series requirements
    if time_index_col is None:
        raise ValueError(
            "If launching time series regime, time_index_col should be specidfied."
        )
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "If launching time series regime, training data should be pandas DataFrame."
        )

    sorted_idx = np.argsort(X[time_index_col].values)
    y = convert_to_numpy(y)[sorted_idx]
    X = X.iloc[sorted_idx].drop(columns=time_index_col).reset_index(drop=True)

    return X, y
