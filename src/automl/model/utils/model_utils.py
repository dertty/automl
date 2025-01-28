import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


def get_splitter(model_type, n_splits: int = 5, time_series: bool = False, random_state: int = 0):
    if model_type == 'classification':
        return TimeSeriesSplit(n_splits=n_splits) if time_series else \
            StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    elif model_type == 'regression':
        return TimeSeriesSplit(n_splits=n_splits) if time_series else \
            KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    else:
        raise ValueError("Invalid model_type. Use 'classification' or 'regression'.")
    

def get_epmty_array(n: int, n_classes: int | None = None):
    return np.full((n, n_classes), fill_value=np.nan) if n_classes else np.full((n,), fill_value=np.nan)

def get_zeros_array(n: int, n_classes: int | None = None):
    return np.zeros((n, n_classes)) if n_classes else np.full((n,), fill_value=np.nan)