# test_object_columns_selector.py

import pandas as pd
import pytest

from automl.feature_selection.Preprocessing import ObjectColumnsSelector
from automl.feature_selection.Preprocessing import get_array_type, check_array_type


def test_ohe_selection():
    selector = ObjectColumnsSelector(ohe_limiter=2, mode='ohe')
    X = pd.DataFrame({
        'A': ['apple', 'banana', 'cherry'],
        'B': ['dog', 'dog', 'cat'],
        'C': [1, 2, 3],  # Non-object column
        'D': ['red', 'red', 'blue']
    })
    selected_features = selector(X)
    assert selected_features == ['B', 'D']


def test_mte_selection():
    selector = ObjectColumnsSelector(ohe_limiter=2, mode='mte')
    X = pd.DataFrame({
        'A': ['apple', 'banana', 'cherry'],  # 3 unique values
        'B': ['dog', 'dog', 'cat'],          # 2 unique values
        'C': [1, 2, 3],                      # Non-object column
        'D': ['red', 'blue', 'blue']         # 2 unique values
    })
    selected_features = selector(X)
    # For mode 'mte', 'ohe' limit of 2 means selecting columns with >2 unique values
    assert selected_features == ['A']


def test_invalid_mode():
    with pytest.raises(ValueError, match="Mode must be either 'ohe' or 'mte'."):
        selector = ObjectColumnsSelector(ohe_limiter=3, mode='invalid_mode')
        X = pd.DataFrame({
            'A': ['apple', 'banana', 'cherry'],
            'B': ['dog', 'dog', 'cat'],
            'C': [1, 2, 3],
            'D': ['red', 'red', 'blue']
        })
        _ = selector(X)


def test_no_object_columns():
    selector = ObjectColumnsSelector(ohe_limiter=3, mode='ohe')
    X = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    selected_features = selector(X)
    assert selected_features == []


def test_unchanged_dataframe():
    # Test that verifies the original DataFrame is unchanged after selection
    selector = ObjectColumnsSelector(ohe_limiter=3, mode='ohe')
    original_data = {
        'A': ['apple', 'banana', 'cherry'],
        'B': ['dog', 'dog', 'cat'],
        'C': [1, 2, 3],
        'D': ['red', 'red', 'blue']
    }
    X = pd.DataFrame(original_data)
    X_copy = X.copy(deep=True)  # Make a deep copy for comparison
    
    selector(X)
    
    # Verify the DataFrame has not been altered
    pd.testing.assert_frame_equal(X, X_copy)
    

def test_empty_dataframe():
    # Test with an empty DataFrame
    selector = ObjectColumnsSelector(ohe_limiter=3, mode='ohe')
    X = pd.DataFrame()
    selected_features = selector(X)
    assert selected_features == []


def test_dataframe_with_nans():
    # Test with a DataFrame containing NaN values
    selector = ObjectColumnsSelector(ohe_limiter=2, mode='ohe')
    X = pd.DataFrame({
        'A': ['apple', 'banana', 'cat', None],
        'B': [None, 'dog', 'cat', 'cat'],
        'C': [1, None, 3, 2],
        'D': ['red', None, 'red', 'cat']
    })
    selected_features = selector(X)
    # Since the presence of NaN should not affect the object-type recognition,
    # check if the columns are selected based on the count of unique non-null values.
    assert selected_features == ['B', 'D']
