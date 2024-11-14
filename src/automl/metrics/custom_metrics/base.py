from sklearn.metrics import make_scorer

from ..functions import ScorerWrapper


class BaseMetric:
    def __init__(self):

        # IMPORTANT to set this attribute
        # comparison between models is performed based on it
        self.needs_proba = False
        self.greater_is_better = True
        self.is_has_thr = False
        self.model_type = None
        self.thr = None

    def __call__(self, **kwargs) -> float:
        raise NotImplementedError

    def _get_model_score_name(self) -> str:
        raise NotImplementedError

    def set_thr(self, thr) -> None:
        self.thr = thr

    def get_thr(self) -> float:
        return self.thr

    @property
    def score_name(self) -> str:
        raise NotImplementedError

    @property
    def response_method(self):
        if self.needs_proba:
            return "predict_proba"
        return "predict"

    def _get_scorer(self):
        return make_scorer(
            self,
            response_method=self.response_method,
            greater_is_better=self.greater_is_better,
        )

    def get_scorer(self):
        return ScorerWrapper(
            self._get_scorer(),
            greater_is_better=self.greater_is_better,
            metric_name=self.score_name,
        )
