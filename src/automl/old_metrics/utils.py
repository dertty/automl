import re
import numpy as np

import sber_sota.metrics.metrics as metric_funcs
from sber_sota.metrics.stability_metrics import bootstrap_metric, time_stability_metric, time_stability_plot


def str_to_num(value):
    value = value.replace(',', '.')
    if '.' in value:
        return float(value)
    else:
        return int(value)


def get_postfix(postfix):
    if postfix is not None:
        if isinstance(postfix, str):
            if len(postfix) > 0:
                postfix = f'_{postfix}'
            else:
                postfix = ''
        else:
            postfix = ''
    else:
        postfix = ''
    return postfix


def get_dataset_postfix(index):
    if index == 0:
        postfix = 'train'
    elif index == 1:
        postfix = 'val'
    else:
        postfix = f'test_{index}'
    return get_postfix(postfix)


def get_metric_func(model_type, metric, metrics_dict):
    for k, vs in metrics_dict.items():
        for v in vs:
            if re.match(f'^{v}$', metric):
                if metric not in ('l1', 'l2', 'l2_root', 'regression_l2'):
                    result = [x for x in re.findall('\d*[.,]?\d*', metric) if len(x) > 0]
                    if len(result) == 1:
                        value = str_to_num(result[0])
                        return getattr(metric_funcs, k)(model_type, value)
                return getattr(metric_funcs, k)(model_type)


def get_metrics_values(y_true, y_pred, dates, model_type, metric_names, stabilty_metrics_names, metrics_dict, postfix='', thr={}):
    scores = {}
    for metric in metric_names:
        metric_class = get_metric_func(model_type, metric.lower(), metrics_dict)
        score_name = metric_class.get_score_name()
        scores[f'{score_name}{postfix}'] = metric_class.score(y_true, y_pred)

        if metric_class.is_has_thr:
            metric_class.set_thr(thr.get(score_name, None))
            score = metric_class.score(y_true, y_pred)
            thr_value = metric_class.get_thr()
            scores[f'{score_name}{postfix}_{round(thr_value, 2)}'] = score
            thr[score_name] = thr_value

        if metric in stabilty_metrics_names:
            bootstrap_mean, bootstrap_std = bootstrap_metric(y_true, y_pred, metric_class.score, n=30)
            if metric_class.is_has_thr:
                scores[f'{score_name}{postfix}_{round(thr_value, 2)}_mean'] = bootstrap_mean
                scores[f'{score_name}{postfix}_{round(thr_value, 2)}_std'] = bootstrap_std
            else:
                scores[f'{score_name}{postfix}_mean'] = bootstrap_mean
                scores[f'{score_name}{postfix}_std'] = bootstrap_std

            if dates is not None:
                mean_in_date, mean_std_in_date, std_in_date = time_stability_metric(y_true, y_pred, dates, metric_class.score, n=30)
                if mean_in_date:
                    if not np.isnan(mean_in_date):
                        if metric_class.is_has_thr:
                            scores[f'{score_name}{postfix}_{round(thr_value, 2)}_time_mean'] = mean_in_date
                            scores[f'{score_name}{postfix}_{round(thr_value, 2)}_mean_time_std'] = mean_std_in_date
                            scores[f'{score_name}{postfix}_{round(thr_value, 2)}_time_std'] = std_in_date
                        else:
                            scores[f'{score_name}{postfix}_time_mean'] = mean_in_date
                            scores[f'{score_name}{postfix}_mean_time_std'] = mean_std_in_date
                            scores[f'{score_name}{postfix}_time_std'] = std_in_date
                        
                        fig = time_stability_plot(y_true, y_pred, dates, metric_class, n=30)
                        fig.write_image(f'{score_name}{postfix}.png')

    return scores, thr