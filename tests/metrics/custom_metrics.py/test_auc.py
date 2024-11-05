import numpy as np
import pytest
from automl.metrics import RocAuc


@pytest.fixture
def setup_roc_auc():
    return RocAuc(), np.array([0, 1, 0, 1])  # y_true for tests

def test_roc_auc_binary_probabilities(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred_proba = np.array([[0.5, 0.5],
                              [0.2, 0.8],
                              [0.9, 0.1],
                              [0.2, 0.8]])
    score = roc_auc(y_true, y_pred_proba)
    assert score > 0.5, f'{score} > 0.5'  # Expecting a non-trivial score

def test_roc_auc_invalid_binary_probabilities(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred = np.array([[0.5, 0.5],
                       [0.5, 0.5],
                       [0.5, 0.5],
                       [0.5, 0.5]])
    score = roc_auc(y_true, y_pred)
    assert score == pytest.approx(0.5)  # Expecting a score of 0.5 due to tied predictions

def test_roc_auc_nan_in_predictions(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred = np.array([[0.1, 0.9],
                       [np.nan, 0.8],
                       [0.9, 0.1],
                       [0.6, 0.4]])
    score = roc_auc(y_true, y_pred)
    assert score is None  # Should return None for NaN

def test_roc_auc_error_for_non_probability(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred = np.array([0, 1, 0, 1.1])  # Not probabilities
    with pytest.raises(ValueError, match="Predictions should contain probabilities for metric RocAuc."):
        roc_auc(y_true, y_pred)  # Should raise an error