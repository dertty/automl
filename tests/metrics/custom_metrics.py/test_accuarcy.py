import numpy as np
import pytest
from automl.metrics import Accuracy


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
    y_pred = np.array([[0.1], [0.9], [0.1], [0.6]])
    score = accuracy(y_true, y_pred)
    assert score == 1.0