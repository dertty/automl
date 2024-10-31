import numpy as np
import pandas as pd
import pytest

from automl.feature_selection.selectors import NanFeatureSelector
from automl.utils.utils import get_array_type, check_array_type


def test_unchanged_dataframe():
    # Initialize the NanFeatureSelector with the default threshold
    nfs = NanFeatureSelector(nan_share_ts=0.2)
    
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
    nfs(X)
    
    # Check if the DataFrame remains unchanged
    pd.testing.assert_frame_equal(X, X_copy)


def test_default_threshold():
    # Initialize the NanFeatureSelector with default threshold of 0.2
    nfs = NanFeatureSelector()
    
    # Create a DataFrame with the expected ratio of NaNs
    X = pd.DataFrame({
        'A': [1, None, 3, None],   # 50% NaNs
        'B': [1, None, None, None], # 75% NaNs
        'C': [1, 2, 3, 4]          # 0% NaNs
    })
    
    # Calculate the expected selected features
    # Here, we assume 'B' should be selected under the default threshold of 0.2
    selected_features = nfs(X)
    
    # Check that only 'B' is returned
    assert selected_features == ['A', 'B']


def test_custom_threshold():
    # Test with a custom threshold of 0.5
    nfs = NanFeatureSelector(nan_share_ts=0.5)
    X = pd.DataFrame({
        'A': [1, 2, None, None],
        'B': [None, None, None, None],
        'C': [1, None, 1, 1]
    })
    selected_features = nfs(X)
    assert selected_features == ['A', 'B']


def test_all_features():
    # Test where all features should be selected
    nfs = NanFeatureSelector(nan_share_ts=0.1)
    X = pd.DataFrame({
        'A': [None, None, None],
        'B': [None, None, 1],
        'C': [None, 1, 1]
    })
    selected_features = nfs(X)
    assert selected_features == ['A', 'B', 'C']


def test_no_features():
    # Test where no features should be selected
    nfs = NanFeatureSelector(nan_share_ts=1.0)
    X = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [1, 2, 3],
        'C': [1, 2, 3]
    })
    selected_features = nfs(X)
    assert selected_features == []


def test_invalid_input():
    # Test for invalid input type, assuming exceptions are properly raised
    nfs = NanFeatureSelector()
    with pytest.raises(ValueError):
        nfs(42)  # Assuming non-array input
        
