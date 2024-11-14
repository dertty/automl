import numpy as np
import pandas as pd

import sklearn.metrics as metrics


def get_k_for_at_k(k, max_k):
    if isinstance(k, float):
        if k == 1.0:
            k = max_k
        else:
            k = max(1, int(max_k * k))
    else:
        if k > max_k:
            k = max_k
    return k


def list2array(l):
    if isinstance(l, list):
        return np.array(l)
    return l


def lists2arrays(y_true, y_pred):
    return list2array(y_true), list2array(y_pred)


def precision_at_k(y_true, y_pred, k=3):
    y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    k = get_k_for_at_k(k, len(y))
    y = y.sort_values('y_pred', ascending=False)[:k]
    if len(y) == 0:
        return 0
    return y['y_true'].sum() / k


def recall_at_k(y_true, y_pred, k=3):
    y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    k = get_k_for_at_k(k, len(y))
    y = y.sort_values('y_pred', ascending=False)
    if len(y) == 0:
        return 0
    return 0 if y['y_true'].sum() == 0 else y['y_true'][:k].sum() / y['y_true'].sum()


def ap_at_k(y_true, y_pred, k=3):
    y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    k = get_k_for_at_k(k, len(y))
    y = y.sort_values('y_pred', ascending=False)
    y = y['y_true'].values
    return 0 if y[:k].sum() == 0 else sum([(y[:i + 1].sum() / (i + 1)) * y[i] for i in range(k)]) / y[:k].sum()


def map_at_k(y_true, y_pred, k=3):
    y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    k = get_k_for_at_k(k, len(y))
    if k == 0:
        return 0
    y = y.sort_values('y_pred', ascending=False)
    y = y['y_true'].values
    ap_at_k = lambda y, k: 0 if y[:k].sum() == 0 else sum([(y[:i + 1].sum() / (i + 1)) * y[i] for i in range(k)]) / y[:k].sum()
    return sum([ap_at_k(y, i) for i in range(1, k + 1)]) / k


def f_beta(y_true, y_pred, beta=1, thr=None, return_thr=False):
    y_true, y_pred = lists2arrays(y_true, y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0
    y_pred_unique = np.unique(y_pred)
    if (len(y_pred_unique) == 2 and 0 in y_pred_unique and 1 in y_pred_unique)\
        or (len(y_pred_unique) == 1 and (0 in y_pred_unique or 1 in y_pred_unique)):
            return metrics.fbeta_score(y_true, y_pred, beta=beta)
    if thr is None:
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        thresholds = np.insert(thresholds, 0, 0)
        f_beta_scores = np.zeros(len(precisions))
        non_zero_indicies = (precisions + recalls) != 0
        f_beta_scores[non_zero_indicies] = ((1 + beta * beta) * precisions[non_zero_indicies] * recalls[non_zero_indicies]) / (beta * beta * precisions[non_zero_indicies] + recalls[non_zero_indicies])
        ix = np.argmax(f_beta_scores)
        best_fbeta_score = f_beta_scores[ix]
        if not return_thr:
            return best_fbeta_score
        else:
            return best_fbeta_score, thresholds[ix]
    if not return_thr:
        return metrics.fbeta_score(y_true, y_pred >= thr, beta=beta)
    else:
        return metrics.fbeta_score(y_true, y_pred >= thr, beta=beta), thr


def precision(y_true, y_pred, thr=None, return_thr=False):
    y_true, y_pred = lists2arrays(y_true, y_pred)
    if len(y_true) == 0:
        return 0
    y_pred_unique = np.unique(y_pred)
    if (len(y_pred_unique) == 2 and 0 in y_pred_unique and 1 in y_pred_unique)\
        or (len(y_pred_unique) == 1 and (0 in y_pred_unique or 1 in y_pred_unique)):
            return metrics.precision_score(y_true, y_pred)
    if thr is None:
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        thresholds = np.insert(thresholds, 0, 0)
        ix = np.argmax(precisions)
        best_precision_score = precisions[ix]
        if not return_thr:
            return best_precision_score
        else:
            return best_precision_score, thresholds[ix]
    if not return_thr:
        return metrics.precision_score(y_true, y_pred >= thr)
    else:
        return metrics.precision_score(y_true, y_pred >= thr), thr


def recall(y_true, y_pred, thr=None, return_thr=False):
    y_true, y_pred = lists2arrays(y_true, y_pred)
    if len(y_true) == 0:
        return 0
    y_pred_unique = np.unique(y_pred)
    if (len(y_pred_unique) == 2 and 0 in y_pred_unique and 1 in y_pred_unique)\
        or (len(y_pred_unique) == 1 and (0 in y_pred_unique or 1 in y_pred_unique)):
            return metrics.recall_score(y_true, y_pred)
    if thr is None:
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        thresholds = np.insert(thresholds, 0, 0)
        ix = np.argmax(recalls)
        best_recall_score = recalls[ix]
        if not return_thr:
            return best_recall_score
        else:
            return best_recall_score, thresholds[ix]
    if not return_thr:
        return metrics.recall_score(y_true, y_pred >= thr)
    else:
        return metrics.recall_score(y_true, y_pred >= thr), thr


def prauc(y_true, y_pred):
    if len(y_true) == 0:
        return 0
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = lists2arrays(y_true, y_pred)
    return (100 / len(y_true)) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def regression_roc_auc_score(y_true, y_pred, num_rounds = 10000):

    '''
    Computes Regression-ROC-AUC-score.

    Parameters:
    -———
    y_true: array-like of shape (n_samples,). Binary or continuous target variable.
    y_pred: array-like of shape (n_samples,). Target scores.
    num_rounds: int or string. If integer, number of random pairs of observations. 
    If string, ‘exact’, all possible pairs of observations will be evaluated.

    Returns:
    -——
    rroc: float. Regression-ROC-AUC-score.

    '''
    def _yield_pairs(y_true, num_rounds):
        '''
        Returns pairs of valid indices. Indices must belong to observations having different values.

        Parameters:
        -———
        y_true: array-like of shape (n_samples,). Binary or continuous target variable.
        num_rounds: int or string. If integer, number of random pairs of observations to return. 
        If string, ‘exact’, all possible pairs of observations will be returned.

        Yields:
        -——
        i, j: tuple of int of shape (2,). Indices referred to a pair of samples.

        '''
        if num_rounds == 'exact':
            for i in range(len(y_true)):
                for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                    yield i, j 
        else:
            for r in range(num_rounds):
                i = np.random.choice(range(len(y_true)))
                j = np.random.choice(np.where(y_true != y_true[i])[0])
                yield i, j

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1

    roc_auc_score = num_same_sign / num_pairs

    return roc_auc_score
