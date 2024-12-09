from numba import njit
import numpy as np

@njit(cache=True)
def regression_roc_auc_score(y_true, y_pred):
    num_pairs = 0
    num_same_sign = 0
    for i in range(len(y_true)):
        for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
            diff_true = y_true[i] - y_true[j]
            diff_score = y_pred[i] - y_pred[j]
            if diff_true * diff_score > 0:
                num_same_sign += 1
            elif diff_score == 0:
                num_same_sign += .5
            num_pairs += 1
    roc_auc_score = num_same_sign / num_pairs
    return roc_auc_score