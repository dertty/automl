import inspect
from sklearn.metrics import get_scorer_names, get_scorer as sk_get_scorer


class ScorerWrapper:
    def __init__(self, scorer, greater_is_better=True):
        self.scorer = scorer
        self.greater_is_better = greater_is_better

    def __call__(self, estimator, X, y):
        return self.scorer(estimator, X, y)
    
    def score(self, y, y_pred):
        return self.scorer._score_func(y, y_pred)
    
    def is_better(self, val1, val2):
        if val1 is None or val2 is None:
            return False
        return val1 > val2 if self.greater_is_better else val1 < val2


def is_scorer(obj):
    # Проверяем, что объект вызываемый (имеет метод __call__)
    if not callable(obj):
        return False
    
    # Проверяем сигнатуру вызова
    try:
        sig = inspect.signature(obj.__call__)
        params = list(sig.parameters)
        # Проверяем, что в сигнатуре есть три обязательных параметра: estimator, X и y
        return params[:3] == ['estimator', 'y', 'y_pred'] or params[:3] == ['estimator', 'y_true', 'y_pred']
    except (TypeError, ValueError):
        # Если сигнатуру не удалось получить, это не scorer
        return False
    

def is_metric(obj):
    # Проверяем, что объект вызываемый (имеет метод __call__)
    if not callable(obj):
        return False
    
    # Проверяем сигнатуру вызова
    try:
        sig = inspect.signature(obj.__call__)
        params = list(sig.parameters)
        # Проверяем, что в сигнатуре есть два обязательных параметра: X и y
        return params[:2] == ['y', 'y_pred'] or params[:3] == ['y_true', 'y_pred']
    except (TypeError, ValueError):
        # Если сигнатуру не удалось получить, это не scorer
        return False


def get_scorer(metric):
    if isinstance(metric, str):
        all_scorers = get_scorer_names()
        if metric in all_scorers:
            scorer = sk_get_scorer(metric)
            return ScorerWrapper(scorer, greater_is_better=True)
        else:
            raise ValueError(f"Metric name '{metric}' is not available. Check name or use custom metric")
    elif is_scorer(metric):
        return metric
    elif is_metric(metric):
        if hasattr(metric, 'get_scorer'):
            return metric.get_scorer()
        else:
            raise AttributeError("The metric object does not have a 'get_scorer' method.")
    else:
        raise ValueError(f"Wrong scorer. Check https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring")