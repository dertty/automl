import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ...loggers import get_logger
from ..base_model import BaseModel, BaseTuner
from ..metrics import MSE
from ..type_hints import FeaturesType, TargetType
from ..utils import optuna_tune, convert_to_numpy, convert_to_pandas


log = get_logger(__name__)


class LightGBMClassification(BaseModel):
    def __init__(
        self,
        objective_type="binary",
        boosting="gbdt",
        num_iterations=1_000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=20,
        bagging_fraction=1,
        bagging_freq=0,
        feature_fraction=1,
        early_stopping_round=100,
        lambda_l1=0,
        lambda_l2=0,
        min_gain_to_split=0,
        is_unbalance=False,
        class_weight=None,
        n_jobs=6,
        random_state=42,
        time_series=False,
    ):

        self.name = "LightGBMClassification"

        # Гиперпараметры
        self.objective_type = objective_type
        self.boosting = boosting
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.feature_fraction = feature_fraction
        self.early_stopping_round = early_stopping_round
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.num_leaves = num_leaves
        self.is_unbalance = is_unbalance
        self.class_weight = class_weight

        # Прочие параметры
        self.num_threads = n_jobs
        self.random_state = random_state
        self.verbose = -1
        
        # Параметры датасета
        self.categorical_feature = []
        self.time_series = time_series
        self.splitter = lambda n_splits: TimeSeriesSplit(n_splits=n_splits) if self.time_series else StratifiedKFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)

    def fit(self, X: FeaturesType, y: TargetType, categorical_features: list[str] = [], n_splits: int = 5):
        log.info(f"Fitting {self.name}", msg_type="start")

        self.categorical_feature = categorical_features

        X = convert_to_pandas(X)
        y = convert_to_numpy(y)
        y = y.reshape(y.shape[0])
        self.n_classes = np.unique(y).shape[0]

        # correct objective based on the number of classes
        if self.n_classes > 2:
            self.objective_type = "multiclass"

        cv = self.splitter(n_splits).split(X, y)

        self.models = []
        oof_preds = np.full((y.shape[0], self.n_classes), fill_value=np.nan)
        for i, (train_idx, test_idx) in enumerate(cv):
            log.info(f"{self.name} fold {i}", msg_type="fit")

            params = self.params

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

            # fit/predict fold model
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[test_idx], y[test_idx])],
                verbose=False,
                early_stopping_rounds=params['early_stopping_round'],
                categorical_feature=self.categorical_feature,
            )
            oof_preds[test_idx] = fold_model.predict_proba(X.iloc[test_idx])

            # append fold model
            self.models.append(fold_model)

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds
    
    def _predict(self, X_test):
        """Predict on one dataset. Average all fold models"""
        X_test = convert_to_pandas(X_test)

        y_pred = np.zeros((X_test.shape[0], self.n_classes))
        for fold_model in self.models:
            y_pred += fold_model.predict_proba(X_test)

        y_pred = y_pred / len(self.models)
        return y_pred

    @property
    def params(self):
        params = {
            "boosting": "gbdt",
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "objective_type": self.objective_type,
            "num_classes": 1 if self.n_classes == 2 else self.n_classes,
            "verbose": self.verbose,
            "early_stopping_round": self.early_stopping_round,
            "num_threads": self.num_threads,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "feature_fraction": self.feature_fraction,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "is_unbalance": self.is_unbalance,
            "class_weight": self.class_weight
        }

        return params


class LightGBMClassificationTuner(BaseTuner):
    
    def get_trial_params_stage_1(self, trial: optuna.Trial) -> dict:
        match self.mode:
            case 'fast':
                pass
            case 'balanced':
                pass
            case _:
                pass
        raise NotImplementedError
    
    def tune(self):
        optuna_early_stopping_rounds = 20 if self.mode == 'fast' else 50 if self.mode == 'balanced' else 100
        params = {
            'name': self.model_params['name'], 
            'objective': self.objective, 
            'early_stopping_rounds': optuna_early_stopping_rounds, 
            'random_state': self.model_params['random_state'],
        }
        study_stage_1 = optuna_tune(trial_params_func=self.get_trial_params_stage_1, **params)
        
        study_stage_2 = optuna_tune(trial_params_func=self.get_trial_params_stage_2, **params)

        study_stage_3 = optuna_tune(trial_params_func=self.get_trial_params_stage_3, **params)

        study_stage_4 = optuna_tune(trial_params_func=self.get_trial_params_stage_4, **params)

        # set best parameters
        for key, val in study.best_params.items():
            setattr(self.model, key, val)
            
        return self.model
    