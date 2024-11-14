import numpy as np
import pandas as pd
from sklearn.utils import resample
import plotly.express as px


def bootstrap_metric(y_true, y_pred, metric_func, n=30):
    metrics = []
    for i in range(n):
        y_true_resampled, y_pred_resampled = resample(y_true, y_pred)
        if len(np.unique(y_true_resampled)) > 1:
            metrics.append(metric_func(y_true_resampled, y_pred_resampled))
    return np.mean(metrics), np.std(metrics)


def _time_stability_metric(y_true, y_pred, dates, metric_func, n=30):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'date': dates})
    res = df.groupby('date').apply(lambda x: pd.Series(bootstrap_metric(x.y_true, x.y_pred, metric_func, n=n))).sort_index().reset_index()
    res.columns = ['date', 'mean', 'std']
    return res


def time_stability_metric(y_true, y_pred, dates, metric_func, n=30):
    res = _time_stability_metric(y_true, y_pred, dates, metric_func, n=n)
    return np.mean(res['mean']), np.std(res['std']), np.std(res['mean'])


def time_stability_plot(y_true, y_pred, dates, metric, n=30):
    res = _time_stability_metric(y_true, y_pred, dates, metric.score, n=n)
    score_name = metric.get_score_name()
    res.columns = ['date', score_name, 'std']
    return px.line(res, x='date', y=score_name, error_y='std')