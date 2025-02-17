import pandas as pd
import numpy as np

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    # check_X,
)

def find_correlated_features(
    X: pd.DataFrame,
    variables,
    method: str,
    threshold: float,
):
    """
    Much faster of way of computing correlation.
    Uses `np.corrcoef` inside. For now, only `spearman` correlation is available.
    """
    # the correlation matrix
    correlated_matrix = np.corrcoef(X[variables].to_numpy(), rowvar=False)

    # the correlated pairs
    correlated_mask = np.triu(np.abs(correlated_matrix), 1) > threshold

    examined = set()
    correlated_groups = list()
    features_to_drop = list()
    correlated_dict = {}
    for i, f_i in enumerate(variables):
        if f_i not in examined:
            examined.add(f_i)
            temp_set = set([f_i])
            for j, f_j in enumerate(variables):
                if f_j not in examined:
                    if correlated_mask[i, j] == 1:
                        examined.add(f_j)
                        features_to_drop.append(f_j)
                        temp_set.add(f_j)
            if len(temp_set) > 1:
                correlated_groups.append(temp_set)
                correlated_dict[f_i] = temp_set.difference({f_i})

    return correlated_groups, features_to_drop, correlated_dict
