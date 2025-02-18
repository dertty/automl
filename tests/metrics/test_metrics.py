import pytest
import numpy as np
# from sklearn.metrics import get_scorer_names, make_scorer
from automl.metrics import get_scorer

from sklearn.metrics import SCORERS

def get_scorer_names():
    return list(SCORERS.keys())

# Получаем все доступные метрики sklearn
available_metrics = [score for score in get_scorer_names() if score not in ['neg_mean_gamma_deviance', 'neg_mean_poisson_deviance',]]

@pytest.mark.parametrize("metric_name", available_metrics)
def test_sklearn_metrics(metric_name):
    """
    Тестируем каждую метрику из доступных метрик sklearn.
    Проверяем, что метрика доступна и может быть использована.
    """
    try:
        scorer = get_scorer(metric_name)
        # Проверяем, что созданный объект является вызываемым
        assert callable(scorer), f"{metric_name} не является вызываемым объектом."
        result = scorer.score([1, 0, 1], [1, 0, 1])
        assert isinstance(result, (int, float, np.integer, np.floating)), f"Результат ({result}) должен быть числом"
        assert hasattr(scorer, 'greater_is_better'), "У экземпляра Scorer должен быть атрибут greater_is_better."
    except Exception as e:
        pytest.fail(f"Тест не прошел для метрики {metric_name}: {e}")