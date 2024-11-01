from typing import List, Self, Union

import joblib
import numpy as np

from ..constants import PATH, create_ml_data_dir
from ..loggers import configure_root_logger, get_logger
from .catboost import CatBoostClassification, CatBoostRegression
from .lama import TabularLama, TabularLamaNN, TabularLamaUtilized
from .lightgbm import LightGBMClassification, LightGBMRegression
from .linear import LogisticRegression, RidgeRegression
from .sklearn_forests import (
    ExtraTreesClassification,
    ExtraTreesRegression,
    RandomForestClassification,
    RandomForestRegression,
)
from .type_hints import FeaturesType, TargetType
from .utils import convert_to_numpy, save_yaml
from .xgboost import XGBClassification, XGBRegression

log = get_logger(__name__)


class AutoML:
    def __init__(
        self,
        task,
        metric,
        time_series=False,
        models_list=None,
        n_jobs: int = 6,
        random_state: int = 42,
        tuning_timeout=60,
    ):

        self.task = task
        self.metric = metric
        self.time_series = time_series
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tuning_timeout = tuning_timeout

        self.models_list = models_list

        # create directory for storing artefacts
        create_ml_data_dir()
        # configure root logger to log in files
        configure_root_logger()

        self.models_list = models_list
        if self.models_list is None:
            # fill the model_list with models
            if self.task == "regression":
                self.models_list = [
                    RidgeRegression(
                        random_state=self.random_state, time_series=self.time_series
                    ),
                    RandomForestRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    ExtraTreesRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    CatBoostRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    LightGBMRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    XGBRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLama(
                        task="regression",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaUtilized(
                        task="regression",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="mlp",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="denselight",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="dense",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="resnet",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="node",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="autoint",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="regression",
                        nn_name="fttransformer",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                ]
            elif self.task == "classification":
                self.models_list = [
                    LogisticRegression(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    RandomForestClassification(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    ExtraTreesClassification(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    CatBoostClassification(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    LightGBMClassification(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    XGBClassification(
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLama(
                        task="classification",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaUtilized(
                        task="classification",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="mlp",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="denselight",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="dense",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="resnet",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="node",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="autoint",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                    TabularLamaNN(
                        task="classification",
                        nn_name="fttransformer",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        time_series=self.time_series,
                    ),
                ]
            else:
                raise AttributeError(
                    f"Task type '{self.task}' is not supported. Available tasks: 'regression', 'classification'."
                )

        self.path = PATH
        self.best_score = None

    def fit(
        self,
        X: FeaturesType,
        y: TargetType,
        Xs_test: Union[FeaturesType, List[FeaturesType]],
        ys_test: Union[TargetType, List[TargetType]],
        categorical_features=[],
    ) -> Self:
        """If self.time_series == True -> X should be sorted by time."""

        self.feature_names = X.columns
        y = convert_to_numpy(y)

        for i, model in enumerate(self.models_list):
            log.info(
                f"{i + 1} out of {len(self.models_list)}. {model.name}",
                msg_type="model",
            )

            log.info(f"Working with {model.name}", msg_type="start")

            # tune the model
            model.tune(
                X,
                y,
                metric=self.metric,
                timeout=self.tuning_timeout,
                categorical_features=categorical_features,
            )

            # fit the tuned model and predict on train
            oof_preds = model.fit(X, y, categorical_features=categorical_features)
            ys_trian = model.predict(X)

            # evaluate on train
            train_scores = self.evaluate(y, ys_trian)
            log.info(f"Train: {train_scores}", msg_type="score")

            # evaluate on out_of_fold
            # remove possible Nones in oof
            if oof_preds.ndim == 1:
                # regression
                not_none_oof = np.where(np.logical_not(np.isnan(oof_preds)))[0]
            else:
                # classification
                not_none_oof = np.where(np.logical_not(np.isnan(oof_preds[:, 0])))[0]

            oof_scores = self.evaluate(y[not_none_oof], oof_preds[not_none_oof])
            log.info(f"OOF: {oof_scores}", msg_type="score")

            # predict on test and evaluate the model
            ys_pred = model.predict(Xs_test)
            test_scores = self.evaluate(ys_test, ys_pred)
            log.info(f"Test: {test_scores}", msg_type="score")
            log.info(
                f"Overfit: {(abs(test_scores - train_scores) / train_scores) * 100 :.2f} %",
                msg_type="score",
            )

            # create model's directory
            model_dir = self.path / model.name
            model_dir.mkdir(exist_ok=True)

            # save the model
            joblib.dump(model, model_dir / f"{model.name}.joblib")

            # save best model's parameters
            save_yaml(model.params, model_dir / f"{model.name}.yaml")

            log.info(f"Working with {model.name}", msg_type="end")

            # compare current model with the best model
            # if metric is better -> new best model
            if i == 0 or self.metric.is_better(test_scores, self.best_score):
                self.best_model = model
                self.best_score = test_scores
                log.info(
                    f"{self.best_model.name}. Best score: {self.best_score} \n",
                    msg_type="new_best",
                )
            else:
                log.info(
                    f"{self.best_model.name}. Best score: {self.best_score} \n",
                    msg_type="best",
                )
        return self

    def predict(self, X: FeaturesType, model_name=None):

        # inference the model
        y_pred = self.best_model.predict(X)

        return y_pred

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        return self.metric(y_true, y_pred)
