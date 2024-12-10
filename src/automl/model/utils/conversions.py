import numpy as np
import pandas as pd


def convert_to_numpy(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float32)

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, list):
        return np.array(data, dtype=np.float32)

    raise AttributeError(
        "Input data is of incorrect type. Supported types: 'pandas' ,'numpy'"
    )


def convert_to_pandas(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data

    if isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=[f"column_{i}" for i in range(data.shape[1])])

    raise AttributeError(
        "Input data is of incorrect type. Supported types: 'pandas' ,'numpy'"
    )
