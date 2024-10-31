import numpy as np
import pandas as pd
import pytest

from automl.feature_selection.selectors import QConstantFeatureSelector
from automl.utils.utils import get_array_type, check_array_type



def test_unchanged_dataframe():
    # Initialize the NanFeatureSelector with the default threshold
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.98)
    
    # Create a test DataFrame
    original_data = {
        'A': [1, None, 3, None],
        'B': [1, None, None, None],
        'C': [1, 2, 3, 4]
    }
    X = pd.DataFrame(original_data)
    
    # Make a copy of the original DataFrame to compare later
    X_copy = X.copy(deep=True)
    
    # Apply the feature selector
    qfs(X)
    
    # Check if the DataFrame remains unchanged
    pd.testing.assert_frame_equal(X, X_copy)


def test_constant_features():
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.98)
    X = pd.DataFrame({
        'A': [1, 1, 1, 1],    # 100% same
        'B': [1, 2, 1, 1],    # 75% same
        'C': [3, 3, 3, 4]     # 75% same
    })
    selected_features = qfs(X)
    assert selected_features == ['A']


def test_quasi_constant_features():
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.75)
    X = pd.DataFrame({
        'A': [1, 1, 1, 1],   # 100% same
        'B': [1, 2, 1, 1],   # 75% same
        'C': [3, 3, 3, 4]    # 75% same
    })
    selected_features = qfs(X)
    assert selected_features == ['A', 'B', 'C']


def test_quasi_constant_with_high_threshold():
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.99)
    X = pd.DataFrame({
        'A': [1, 1, 1, 2],   # 75% same
        'B': [1, 1, 1, 1],   # 100% same
        'C': [4, 4, 4, 4]    # 100% same
    })
    selected_features = qfs(X)
    assert selected_features == ['B', 'C']


def test_no_constant_features():
    # Test when there are no constant or quasi-constant features
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.98)
    X = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [1, 2, 3, 4],
        'C': [4, 3, 2, 1]
    })
    selected_features = qfs(X)
    assert selected_features == []


def test_invalid_input_type():
    # Test for invalid input type
    qfs = QConstantFeatureSelector(feature_val_share_ts=0.98)
    with pytest.raises(ValueError):
        qfs(42)  # Assuming non-array input raises an error
