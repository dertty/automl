import warnings

import numpy as np
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import TimeSeriesIterator
from torch import set_num_threads as set_num_threads_torch

from ...loggers import get_logger
from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import convert_to_pandas

warnings.filterwarnings("ignore")


log = get_logger(__name__)


class TabularLamaNN(BaseModel):
    def __init__(
        self,
        categorical_features=None,
        nn_name="mlp",
        timeout=60,
        task="regression",
        n_jobs=6,
        random_state=42,
        n_folds=5,
        scorer=None,
        time_series=False,
    ):

        self.name = f"TabularLamaNN_{nn_name}"

        self.categorical_features = []
        self.nn_name = nn_name
        self.timeout = timeout
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        self.scorer = scorer
        self.time_series = time_series

        if task == "regression":
            self.task = "reg"
        else:
            self.task = task

        self.verbose = 1

        set_num_threads_torch(self.n_jobs)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features=[]):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_features = categorical_features
        np.random.seed(self.random_state)

        X = convert_to_pandas(X)
        data = X.assign(target=y)

        # adjust task in the case of classification
        if self.task == "classification":
            if np.unique(y).shape[0] == 2:
                self.task = "binary"
            else:
                self.task = "multiclass"

        model = TabularAutoML(
            task=Task(
                name=self.task,
                metric=self.scorer.score,
                greater_is_better=self.scorer.greater_is_better,
            ),
            general_params={"use_algos": [[self.nn_name]]},
            timeout=2 * self.timeout,
            cpu_limit=self.n_jobs,
            nn_pipeline_params={"use_qnt": True, "use_te": False},
            reader_params={"n_jobs": 1, "cv": 5, "random_state": self.random_state},
        )

        roles = {"target": "target"}
        if len(self.categorical_features) > 0:
            roles["category"] = self.categorical_features

        if self.time_series:
            artificial_index = np.arange(X.shape[0])
            oof_preds = model.fit_predict(
                data,
                roles=roles,
                verbose=self.verbose,
                cv_iter=TimeSeriesIterator(artificial_index),
            ).data
        else:
            oof_preds = model.fit_predict(data, roles=roles, verbose=self.verbose).data

        # flatten the output in regression case
        # and add 0 class probabilities in binary case
        if self.task == "reg":
            oof_preds = oof_preds.reshape(oof_preds.shape[0])
        elif self.task == "binary":
            oof_preds = np.hstack((1 - oof_preds, oof_preds))

        self.model = model

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds

    def tune(
        self,
        X: FeaturesType,
        y: TargetType,
        scorer=None,
        timeout=None,
        categorical_features=[],
    ):
        self.timeout = timeout
        self.scorer = scorer
        self.categorical_features = categorical_features

    def _predict(self, X_test):
        X_test = convert_to_pandas(X_test)
        y_pred = self.model.predict(X_test).data

        # flatten the output in regression case
        # and add 0 class probabilities in binary case
        if self.task == "reg":
            y_pred = y_pred.reshape(y_pred.shape[0])
        elif self.task == "binary":
            y_pred = np.hstack((1 - y_pred, y_pred))

        return y_pred

    @property
    def params(self):
        return {
            "nn_name": self.nn_name,
            "timeout": self.timeout,
            "task": self.task,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "n_folds": self.n_folds,
            "verbose": self.verbose,
        }
