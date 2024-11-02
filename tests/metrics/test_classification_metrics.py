import numpy as np
import pytest
from automl.metrics import Accuracy, RocAuc  # Replace `your_module` with the appropriate module name


@pytest.fixture
def setup_accuracy():
    return Accuracy(thr=0.5), np.array([0, 1, 0, 1])  # y_true for tests


def test_accuracy_label_predictions(setup_accuracy):
    accuracy, y_true = setup_accuracy
    y_pred = np.array([0, 1, 0, 1])
    score = accuracy(y_true, y_pred)
    assert score == 1.0


def test_accuracy_prob_predictions(setup_accuracy):
    accuracy, y_true = setup_accuracy
    y_pred = np.array([0., 1., 0.49, 0.51])
    score = accuracy(y_true, y_pred)
    assert score == 1.0


def test_accuracy_probabilities(setup_accuracy):
    accuracy, y_true = setup_accuracy
    # [1, 1, 0, 0]
    y_pred_proba = np.array([[0.1, 0.9],
                              [0.2, 0.8],
                              [0.9, 0.1],
                              [0.6, 0.4]])
    score = accuracy(y_true, y_pred_proba)
    assert score == pytest.approx(0.5)


def test_accuracy_nan_in_predictions(setup_accuracy):
    accuracy, y_true = setup_accuracy
    y_pred = np.array([0, np.nan, 0, 1])
    score = accuracy(y_true, y_pred)
    assert score is None  # Should return None for NaN


def test_accuracy_invalid_shape(setup_accuracy):
    accuracy, y_true = setup_accuracy
    # [0, 1, 0, 1]
    y_pred = np.array([[0.1], [0.9], [0.1], [0.5]])
    score = accuracy(y_true, y_pred)
    assert score == 1.0 


@pytest.fixture
def setup_roc_auc():
    return RocAuc(), np.array([0, 1, 0, 1])  # y_true for tests

def test_roc_auc_binary_probabilities(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred_proba = np.array([[0.1, 0.9],
                              [0.2, 0.8],
                              [0.9, 0.1],
                              [0.6, 0.4]])
    score = roc_auc(y_true, y_pred_proba)
    assert score > 0.5  # Expecting a non-trivial score

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

def test_roc_auc_error_for_labels(setup_roc_auc):
    roc_auc, y_true = setup_roc_auc
    y_pred = np.array([0, 1, 0, 1])  # Not probabilities
    with pytest.raises(ValueError, match="Predictions should contain probabilities for metric RocAuc."):
        roc_auc(y_true, y_pred)  # Should raise an error