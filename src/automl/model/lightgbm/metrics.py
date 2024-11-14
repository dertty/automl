def get_eval_metric(scorer):
    """Full metrics list for lightgbm see here:
    https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
    """
    score_name = scorer.get_score_name()
    if score_name.lower() in ["roc_auc", "auc", "rocauc"]:
        return "auc"
    elif score_name.lower() in ["accuracy"]:
        return "binary_error"
    else:
        raise NotImplementedError
    # elif score_name.lower() in get_scorer_names():
    #     return None
    # else:
    #     return get_custom_catboost_metric(scorer.score, scorer.greater_is_better)
