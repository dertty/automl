import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_classes=2, class_sep=2, 
        n_samples=100, n_features=10, random_state=42)
    return X, y

@pytest.fixture
def sample_unbalanced_data():
    X, y = make_classification(
        n_classes=2, class_sep=2, 
        weights=[0.95, 0.05], # 90% одного класса, 10% другого
        n_samples=100, n_features=10, random_state=42)
    return X, y