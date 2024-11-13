import numpy as np
import lightgbm as lgb
import optuna
from typing import Callable, Tuple, Dict

from ..base_model import BaseModel
from ..type_hints import FeaturesType, TargetType


class Tuner(BaseModel):
    
    @staticmethod
    def get_trial_params(trial: optuna.Trial, dataset_shape: Tuple[int, int], mode: str = 'fast') -> dict:
        raise NotImplementedError

    def objective_stage_1(self, trial: optuna.Trial, X: FeaturesType, y: TargetType, splitter, metric_func: Callable, mode: str='fast') -> float:
            trial_params = self.get_trial_params(trial, mode=mode, dataset_shape=X.shape)
            params = {**self.params, **trial_params}
            # BUG LightGBM produces very annoying alias warning.
            # Temp fix is to rename all the input parameters.
            # In the recent versions of lightgbm the issue is solved.
            params["colsample_bytree"] = params.pop("feature_fraction")
            params["reg_alpha"] = params.pop("lambda_l1")
            params["subsample"] = params.pop("bagging_fraction")
            params["min_child_samples"] = params.pop("min_data_in_leaf")
            params["min_split_gain"] = params.pop("min_gain_to_split")
            params["subsample_freq"] = params.pop("bagging_freq")
            params["reg_lambda"] = params.pop("lambda_l2")
            params["n_jobs"] = params.pop("num_threads")
            params["boosting_type"] = params.pop("boosting")

            # add parameter `class_weight` for multiclass only
            if self.n_classes > 2:
                trial_params["class_weight"] = trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )

            cv_metrics = []
            best_num_iterations = []
            for train_idx, test_idx in splitter.split(X, y):
                # train_data = lgb.Dataset(
                #     X[train_idx], y[train_idx], categorical_feature=self.categorical_feature
                # )
                # test_data = lgb.Dataset(
                #     X[test_idx], y[test_idx], categorical_feature=self.categorical_feature
                # )

                # model = lgb.train(
                #     params={**trial_params, **not_tuned_params},
                #     train_set=train_data,
                #     valid_sets=[test_data],
                #     verbose_eval=False
                # )

                # Here I decided to use sklearn API
                # because `lgb.train` does not contain parameter `class_weight`
                # that drastically improves the performance of models
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X.iloc[train_idx],
                    y[train_idx],
                    eval_set=[(X.iloc[test_idx], y[test_idx])],
                    verbose=False,
                    early_stopping_rounds=params["early_stopping_round"],
                    categorical_feature=self.categorical_feature,
                )

                y_pred = model.predict_proba(X.iloc[test_idx])

                cv_metrics.append(metric_func(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)

            # add `num_iterations` to the optuna parameters
            trial.set_user_attr("num_iterations", round(np.mean(best_num_iterations)))

            return np.mean(cv_metrics)
        
    def objective_stage_2(self, trial: optuna.Trial, X: FeaturesType, y: TargetType, splitter, metric_func: Callable, mode: str='fast') -> float:
            trial_params = self.get_trial_params(trial, mode=mode, dataset_shape=X.shape)
            params = {**self.params, **trial_params}
            # BUG LightGBM produces very annoying alias warning.
            # Temp fix is to rename all the input parameters.
            # In the recent versions of lightgbm the issue is solved.
            params["colsample_bytree"] = params.pop("feature_fraction")
            params["reg_alpha"] = params.pop("lambda_l1")
            params["subsample"] = params.pop("bagging_fraction")
            params["min_child_samples"] = params.pop("min_data_in_leaf")
            params["min_split_gain"] = params.pop("min_gain_to_split")
            params["subsample_freq"] = params.pop("bagging_freq")
            params["reg_lambda"] = params.pop("lambda_l2")
            params["n_jobs"] = params.pop("num_threads")
            params["boosting_type"] = params.pop("boosting")

            # add parameter `class_weight` for multiclass only
            if self.n_classes > 2:
                trial_params["class_weight"] = trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )

            cv_metrics = []
            best_num_iterations = []
            for train_idx, test_idx in splitter.split(X, y):
                # train_data = lgb.Dataset(
                #     X[train_idx], y[train_idx], categorical_feature=self.categorical_feature
                # )
                # test_data = lgb.Dataset(
                #     X[test_idx], y[test_idx], categorical_feature=self.categorical_feature
                # )

                # model = lgb.train(
                #     params={**trial_params, **not_tuned_params},
                #     train_set=train_data,
                #     valid_sets=[test_data],
                #     verbose_eval=False
                # )

                # Here I decided to use sklearn API
                # because `lgb.train` does not contain parameter `class_weight`
                # that drastically improves the performance of models
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X.iloc[train_idx],
                    y[train_idx],
                    eval_set=[(X.iloc[test_idx], y[test_idx])],
                    verbose=False,
                    early_stopping_rounds=params["early_stopping_round"],
                    categorical_feature=self.categorical_feature,
                )

                y_pred = model.predict_proba(X.iloc[test_idx])

                cv_metrics.append(metric_func(y[test_idx], y_pred))
                best_num_iterations.append(model.best_iteration_)

            # add `num_iterations` to the optuna parameters
            trial.set_user_attr("num_iterations", round(np.mean(best_num_iterations)))

            return np.mean(cv_metrics)