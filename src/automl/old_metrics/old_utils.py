import numpy as np
import inspect
import .custom_metrics as custom_metrics


def get_catboost_custom_scorer(score_func):
    class CustomMetric:
        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            y_pred = np.array(approxes[0]).astype(float)
            y_true = np.array(target).astype(int)

            output_weight = 1 # weight is not used

            score = score_func(y_true, y_pred)

            return score, output_weight

        def get_final_error(self, error, weight):
            return error
    return CustomMetric()


def parse_k(metric, metric_prefix):
    if metric == metric_prefix:
        k = 1.0
    else:
        k = metric.split('_')[-1]
        if '.' in k:
            k = float(k)
        else:
            k = int(k)
    return k


def get_scorer(type, score_func, **args):
    for param, value in args.items():
        assert value is not None, 'Параметр не должен быть None'
    if type == 'sklearn':
        return make_scorer(lambda y_true, y_pred: score_func(y_true, y_pred, **args)[1], needs_proba=True, greater_is_better=True)
    elif type == 'lightgbm':
        return lambda y_pred, y_true: score_func(y_true.get_label(), y_pred, **args)
    elif type == 'catboost':
        return get_catboost_custom_scorer(lambda y_true, y_pred: score_func(y_true, y_pred, **args)[1])


def get_custom_scorer(metric, scorer=False, type='sklearn', thr=None, return_thr=False):
    if metric in ('symmetric_mean_absolute_percentage_error', 'mean_squared_error', 'prauc', 'recall', 'precision'):
        score_func = getattr(custom_metrics, metric)
    else:
        for metric_prefix, func_name in (
            ('map', 'map_at_k', ),
            ('mean_average_precision', 'map_at_k', ),
            ('ap', 'ap_at_k', ),
            ('average_precision', 'ap_at_k', ),
            ('precision_at_', 'p_at_k', ),
            ('p_at_','p_at_k'),
            ('recall_at_', 'recall_at_k',),
            ):
            if metric.startswith(metric_prefix) and metric not in ('mape', ):
                k = parse_k(metric, metric_prefix)
                score_func = getattr(custom_metrics, func_name)

        for metric_prefix, func_name in (
            ('f_beta', 'f_beta', ),
            ):
            if metric.startswith(metric_prefix):
                beta = parse_k(metric, metric_prefix)
                score_func = getattr(custom_metrics, func_name)
        
    parameters = [param for param in inspect.signature(score_func).parameters.keys() if x not in [y_true, y_pred]]
    args = {}
    if 'beta' in parameters:
        args['beta'] = beta
    if 'thr' in parameters:
        args['thr'] = thr
    if 'return_thr' in parameters:
        args['return_thr'] = return_thr
    if 'k' in parameters:
        args['k'] = k
    
    if scorer:
        return get_scorer(type=type, score_func=score_func, **args)
    else:
        result = lambda y_true, y_pred: score_func(y_true, y_pred, **args)
        if len(result) == 3:
            return result[1]
        else:
            return result[1], result[3]

