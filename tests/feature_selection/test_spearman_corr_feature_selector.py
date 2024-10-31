# test_spearman_corr_feature_selector.py

import numpy as np
import pandas as pd
import pytest

from automl.feature_selection.selectors import SpearmanCorrFeatureSelector
from automl.utils.utils import get_array_type, check_array_type


def test_strongly_correlated_features():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.arange(100),
        'D': np.arange(100) * 2  # Strong correlation with C
    })
    selected_features = selector(X)
    assert set(selected_features).intersection(set(['C', 'D']))

def test_weakly_correlated_features():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100) * 0.5 + np.arange(100) * 0.005,
        'C': np.random.rand(100)
    })
    selected_features = selector(X)
    assert selected_features == []

def test_no_numeric_features():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'A': ['dog', 'cat', 'mouse'],
        'B': ['apple', 'banana', 'cherry'],
        'C': ['red', 'blue', 'green']
    })
    selected_features = selector(X)
    assert selected_features == []

def test_empty_dataframe():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame()
    selected_features = selector(X)
    assert selected_features == []

def test_dataframe_with_nans():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'A': [np.nan, 1, 2, 3, np.nan],
        'B': [4, 5, np.nan, np.nan, 8],
        'C': [1, 2, 3, np.nan, np.nan],
        'D': [1, 1, 1, 1, 1]
    })
    selected_features = selector(X)
    assert selected_features == []

def test_single_column():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'A': np.random.rand(100)
    })
    selected_features = selector(X)
    assert selected_features == []

def test_unchanged_dataframe():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    original_data = {
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.arange(100),
        'D': np.arange(100) * 3
    }
    X = pd.DataFrame(original_data)
    X_copy = X.copy(deep=True)
    
    selector(X)
    
    pd.testing.assert_frame_equal(X, X_copy)

def test_high_correlation_threshold():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.95)
    X = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.arange(100),
        'D': np.arange(100) * 1.01  # Very high correlation with C
    })
    selected_features = selector(X)
    assert set(selected_features).intersection(set(['C', 'D']))

def test_mixed_types_single_numeric():
    selector = SpearmanCorrFeatureSelector(corr_ts=0.8)
    X = pd.DataFrame({
        'Numeric': np.random.rand(100),
        'NonNumeric': ['apple', 'banana', 'cherry'] * 33 + ['apple']
    })
    selected_features = selector(X)
    # With only one numeric column, there can be no correlations over the threshold.
    assert selected_features == []
