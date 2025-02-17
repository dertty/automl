import numpy as np
# from sklearn.metrics import get_scorer_names
from sklearn.metrics import SCORERS

def get_scorer_names():
    return list(SCORERS.keys())


METRICS_GREATER_IS_BETTER = {
            # Регрессионные метрики (чем меньше, тем лучше)
            "RMSE": False,
            "MAE": False,
            "Quantile": False,
            "LogLinQuantile": False,
            "MAPE": False,
            "Poisson": False,
            "MultiRMSE": False,
            
            # Классификационные метрики (чем больше, тем лучше)
            "Accuracy": True,
            "BalancedAccuracy": True,
            "AUC": True,
            "F1": True,
            "Precision": True,
            "Recall": True,
            "MCC": True,
            "Logloss": False,
            "CrossEntropy": False,
            "PRAUC": True,
            "Tweedie": False,
            
            # Ранжирование (чем больше, тем лучше)
            "NDCG": True,
            "DCG": True,
            "PFound": True,
            "AverageGain": True,
            "QueryAverage": True,
        }


def get_custom_catboost_metric(score_func, greater_is_better):
    class CustomMetric:
        def is_max_optimal(self):
            return greater_is_better

        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            y_pred = np.array(approxes[0]).astype(float)
            y_true = np.array(target).astype(int)

            output_weight = 1  # weight is not used

            score = score_func(y_true, y_pred)

            return score, output_weight

        def get_final_error(self, error, weight):
            return error

    return CustomMetric()


def get_eval_metric(scorer):
    score_name = scorer.get_score_name()
    if score_name.lower() in ["roc_auc", "auc", "rocauc"]:
        return "AUC"
    elif score_name.lower() in ["accuracy"]:
        return "Accuracy"
    elif score_name.lower() in get_scorer_names():
        return None
    else:
        return get_custom_catboost_metric(scorer.score, scorer.greater_is_better)
