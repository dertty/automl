from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from typing_extensions import Self

from ..constants import PATH, create_ml_data_dir
from ..loggers import get_logger
from ..metrics import get_scorer
from .blender import CoordDescBlender, OptunaBlender
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
from .utils import convert_to_numpy, convert_to_pandas, save_yaml
from .xgboost import XGBClassification, XGBRegression

log = get_logger(__name__)


class AutoML:
    def __init__(
        self,
        task,
        metric,
        time_series=False,
        models_list=None,
        blend=False,
        stack=False,
        n_jobs: int = 6,
        random_state: int = 42,
        tuning_timeout=60,
    ):
        self.task = task
        self.metric = metric
        self.scorer = get_scorer(metric)
        self.blend = blend
        self.stack = stack
        self.time_series = time_series
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tuning_timeout = tuning_timeout

        self.models_list = models_list

        # create directory for storing artefacts
        create_ml_data_dir()

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
                    # TabularLamaUtilized(
                    #     task="classification",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="mlp",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="denselight",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="dense",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="resnet",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="node",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="autoint",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                    # TabularLamaNN(
                    #     task="classification",
                    #     nn_name="fttransformer",
                    #     n_jobs=self.n_jobs,
                    #     random_state=self.random_state,
                    #     time_series=self.time_series,
                    # ),
                ]
            else:
                raise AttributeError(
                    f"Task type '{self.task}' is not supported. Available tasks: 'regression', 'classification'."
                )

        self.flag_stack_is_best = False
        if self.stack:
            if self.task == "classification":
                self.stacker = LightGBMClassification(
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    time_series=self.time_series,
                )
            else:
                self.stacker = LightGBMRegression(
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    time_series=self.time_series,
                )
            self.stacker.name = "Stacker"
            self.models_list.append(self.stacker)

        self.flag_blend_is_best = False
        if self.blend:
            self.blender = CoordDescBlender()
            self.models_list.append(self.blender)

        self.path = PATH
        self.best_score = None

    def fit(
        self,
        X: FeaturesType,
        y: TargetType,
        Xs_test: Union[FeaturesType, List[FeaturesType]] = None,
        ys_test: Union[TargetType, List[TargetType]] = None,
        categorical_features=[],
        save_models=True,
        save_oof=True,
        save_test=True,
    ) -> Self:
        """If self.time_series == True -> X should be sorted by time."""

        self.feature_names = X.columns
        y = convert_to_numpy(y)

        # for blending and stacking
        oofs = []
        test_preds = []

        for i, model in enumerate(self.models_list):
            log.info(
                f"{i + 1} out of {len(self.models_list)}. {model.name}",
                msg_type="model",
            )

            x_train_iter = X
            x_test_iter = Xs_test

            if model.name == "Blender":
                # working with blender
                # prediction of previous models are now features
                x_train_iter = oofs

                if Xs_test is not None:
                    x_test_iter = test_preds

            if model.name == "Stacker":
                # working with stacking
                # initial features + predictions of previous models are now features
                temp = convert_to_numpy(oofs)
                if temp.shape[-1] == 1:
                    # regression
                    x_train_iter = pd.concat(
                        [X, convert_to_pandas(temp[:, :, 0].T)], axis=1
                    )
                elif temp.shape[-1] == 2:
                    # binary classification
                    x_train_iter = pd.concat(
                        [X, convert_to_pandas(temp[:, :, 1].T)], axis=1
                    )
                else:
                    # multiclass classification
                    x_train_iter = pd.concat(
                        [X, convert_to_pandas(np.hstack(temp))], axis=1
                    )

                if Xs_test is not None:
                    temp = convert_to_numpy(test_preds)
                    if temp.shape[-1] == 1:
                        # regression
                        x_test_iter = pd.concat(
                            [Xs_test, convert_to_pandas(temp[:, :, 0].T)], axis=1
                        )
                    elif temp.shape[-1] == 2:
                        # binary classification
                        x_test_iter = pd.concat(
                            [Xs_test, convert_to_pandas(temp[:, :, 1].T)], axis=1
                        )
                    else:
                        # multiclass classification
                        x_test_iter = pd.concat(
                            [Xs_test, convert_to_pandas(np.hstack(temp))], axis=1
                        )

            log.info(f"Working with {model.name}", msg_type="start")

            # tune the model
            model.tune(
                x_train_iter,
                y,
                scorer=self.scorer,
                timeout=self.tuning_timeout,
                categorical_features=categorical_features,
            )
            # fit the tuned model and predict on train
            oof_preds = model.fit(
                x_train_iter, y, categorical_features=categorical_features
            )
            y_trian_preds = model.predict(x_train_iter)

            # evaluate on train
            train_scores = self.evaluate(y, y_trian_preds)
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

            if x_test_iter is not None and ys_test is not None:
                # predict on test and evaluate the model
                y_test_preds = model.predict(x_test_iter)
                test_scores = self.evaluate(ys_test, y_test_preds)
                log.info(f"Test: {test_scores}", msg_type="score")
                log.info(
                    f"Overfit: {(abs(test_scores - train_scores) / train_scores) * 100 :.2f} %",
                    msg_type="score",
                )
            else:
                # No test data given
                # Select the best model based on the oof
                test_scores = oof_scores

            # create model's directory
            model_dir = self.path / model.name
            model_dir.mkdir(exist_ok=True)

            if save_models:
                # save the model
                joblib.dump(model, model_dir / f"{model.name}.joblib")

            if save_oof:
                # save oof predictions
                pd.DataFrame(
                    oof_preds[not_none_oof],
                    columns=[
                        f"{model.name}_pred_{i}" for i in range(oof_preds.shape[1])
                    ],
                ).to_csv(model_dir / f"oof_preds.csv", index=False)

            if save_test and Xs_test is not None and ys_test is not None:
                # save test predictions
                pd.DataFrame(
                    y_test_preds,
                    columns=[
                        f"{model.name}_pred_{i}" for i in range(y_test_preds.shape[1])
                    ],
                ).to_csv(model_dir / f"test_preds.csv", index=False)

            # save best model's parameters
            save_yaml(model.params, model_dir / f"{model.name}.yaml")

            log.info(f"Working with {model.name}", msg_type="end")

            # compare current model with the best model
            # if metric is better -> new best model
            if i == 0 or self.scorer.is_better(test_scores, self.best_score):
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

            if (self.blend or self.stack) and model.name != "Blender":
                # save oof for blending
                oofs.append(oof_preds)

                if Xs_test is not None:
                    test_preds.append(y_test_preds)

        if self.best_model.name == "Blender":
            self.flag_blend_is_best = True

        if self.best_model.name == "Stacker":
            self.flag_stack_is_best = True

        return self

    def predict(self, X: FeaturesType, model_name=None):

        if self.flag_blend_is_best:
            y_pred = []

            # inference models with non-zero blender weights
            non_zero_idx = np.where(self.blender.weights > 0)[0]
            for idx in non_zero_idx:
                y_pred.append(self.models_list[idx].predict(X))

            # inference blender
            y_pred = self.blender.predict(y_pred)

        elif self.flag_stack_is_best:
            y_pred = []

            # inference all the models
            for idx in range(len(self.models_list)):
                if not self.models_list[idx].name in ["Stacker", "Blender"]:
                    y_pred.append(self.models_list[idx].predict(X))

            y_pred = convert_to_numpy(y_pred)
            if y_pred.shape[-1] == 1:
                # regression
                X = pd.concat([X, convert_to_pandas(y_pred[:, :, 0].T)], axis=1)
            elif y_pred.shape[-1] == 2:
                # binary classification
                X = pd.concat([X, convert_to_pandas(y_pred[:, :, 1].T)], axis=1)
            else:
                # multiclass classification
                X = pd.concat([X, convert_to_pandas(np.hstack(y_pred))], axis=1)

            # inference blender
            y_pred = self.stacker.predict(X)

        else:
            # inference the best model
            y_pred = self.best_model.predict(X)

        return y_pred

    def evaluate(self, y_true: TargetType, y_pred: TargetType):
        if self.task == "classification":
            # binary classifiation
            if len(np.unique(y_true)) <= 2:
                return self.scorer.score(y_true, y_pred[:, 1])

        return self.scorer.score(y_true, y_pred)
