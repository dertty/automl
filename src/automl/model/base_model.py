import enum

from automl.metrics import ScorerWrapper
from .type_hints import FeaturesType, TargetType


class BaseModel:
    """
    General structure of a model.

    All models have common `predict` method.

    However, the method `_predct` that actually predicts the underlying model
        should be implemented for each model separately
    """

    def __init__(self):
        pass

    def fit(self, X: FeaturesType, y: TargetType):
        raise NotImplementedError

    def tune(self, X: FeaturesType, y: TargetType, scorer: ScorerWrapper, timeout: int, categorical_features: list[str]=[]):
        raise NotImplementedError

    def _predict(self, X_test):
        raise NotImplementedError

    @property
    def not_tuned_params(self):
        raise NotImplementedError

    @property
    def inner_params(self):
        raise NotImplementedError

    @property
    def meta_params(self):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    def predict(self, Xs):
        ### BUG
        # if isinstance(Xs, FeaturesType):
        # TypeError: Subscripted generics cannot be used with class and instance checks in < python3.10

        if not isinstance(Xs, list) or self.name == "Blender":
            # only one test
            return self._predict(Xs)

        # several tests
        ys_pred = []
        for X_test in Xs:
            y_pred = self._predict(X_test)
            ys_pred.append(y_pred)

        return ys_pred


@enum.unique
class ModelType(enum.Enum):
    BASE_MODEL = 0
    BLENDER = 1
    STACKER = 2
