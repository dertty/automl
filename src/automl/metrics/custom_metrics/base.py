from automl.type_hints import FeaturesType, TargetType
from sklearn.metrics import make_scorer
from ..functions import ScorerWrapper


class BaseMetric:
    def __init__(self):

        # IMPORTANT to set this attribute
        # comparison between models is performed based on it
        self.needs_proba = True
        self.greater_is_better = True
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, **kwargs):
        raise NotImplementedError
    
    def _get_model_score_name(self):
        raise NotImplementedError
    
    def get_score_name(self):
        raise NotImplementedError
    
    def set_thr(self, thr):
        self.thr = thr

    def get_thr(self):
        return self.thr
    
    def _get_scorer(self):
        return make_scorer(self, response_method='predict', greater_is_better=self.greater_is_better)
    
    def get_scorer(self):
        return ScorerWrapper(self._get_scorer(), greater_is_better=self.greater_is_better)