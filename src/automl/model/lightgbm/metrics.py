from typing import Optional, Callable, Tuple
import numpy as np
import lightgbm as lgb
from automl.metrics import ScorerWrapper


METRICS_GREATER_IS_BETTER = {
    'auc': True,
    'average_precision': True,
    'ndcg': True,
    'map': True,
    'binary_logloss': False,
    'categorical_crossentropy': False,
    'cross_entropy': False,
    'l1': False,
    'l2': False,
    'rmse': False,
    'mse': False,
}

def get_custom_lightgbm_metric(score_func: Callable, greater_is_better: bool, metric_name: str = 'custom_metric') -> Callable:
    def custom_metric(preds: np.ndarray, data: lgb.Dataset, score_func: Callable) -> Tuple[str, float, bool]:
        label = data.get_label()
        # weight = data.get_weight()
        score_value = score_func(label, preds)
        return metric_name, score_value, greater_is_better
    return lambda preds, data: custom_metric(preds, data, score_func)


def get_eval_metric(scorer: Optional[ScorerWrapper] = None, score_name: Optional[str] = None):
    """Full metrics list for lightgbm see here:
    https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
    """
    if scorer and score_name:
        raise ValueError("Only one of 'scorer' or 'score_name' should be provided.")
    if score_name:
        return score_name
    if scorer:
        return get_custom_lightgbm_metric(scorer.score, scorer.greater_is_better)
        
