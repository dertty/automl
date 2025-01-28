METRICS_GREATER_IS_BETTER = {
            'auc': True,
            'aucpr': True,
            'map': True,
            'ndcg': True,
            'logloss': False,
            'error': False,
            'rmse': False,
            'mae': False,
            'mape': False,
            'rmsle': False,
        }

def get_eval_metric(scorer):
    """Full metrics list for xgboost see here:
    https://xgboost.readthedocs.io/en/stable/parameter.html
    """
    score_name = scorer.get_score_name()
    if score_name.lower() in ["roc_auc", "auc", "rocauc"]:
        return "auc"
    elif score_name.lower() in ["accuracy"]:
        return "error"
    else:
        raise NotImplementedError
