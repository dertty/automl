from automl.type_hints import FeaturesType, TargetType


class BaseMetric:
    def __init__(self):

        # IMPORTANT to set this attribute
        # comparison between models is performed based on it
        self.needs_proba = True
        self.greater_is_better = True
        self.is_has_thr = False
        self.model_type = None

    def __call__(self, **kwargs):
        raise NotImplementedError
    
    def _get_model_score_name(self):
        return None
    
    def get_score_name(self):
        return ''
    
    def set_thr(self, thr):
        self.thr = thr

    def get_thr(self):
        return self.thr

    def is_better(self, val1, val2):
        if val1 is None or val2 is None:
            return False
        return val1 > val2 if self.greater_is_better else val1 < val2


class BaseScorer:
    """
    Base scorer that accepts the metric that should be scored
    and the prediction function that would be called from the estimator.
    """

    def __init__(self, metric: BaseMetric, pred_func="predict"):
        self.metric = metric
        self.pred_func = pred_func

        # hgher values produced by the scorer indicate better fits
        # multiply results by -1 if lower values of metric are better
        self.sign = -1
        if metric.greater_is_better:
            self.sign = 1

    def __call__(self, estimator, X: FeaturesType, y: TargetType):
        predict = getattr(estimator, self.pred_func)
        y_pred = predict(X)

        return self.sign * self.metric(y, y_pred)
