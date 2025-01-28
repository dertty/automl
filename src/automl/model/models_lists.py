from .catboost import CatBoostClassification, CatBoostRegression
from .lama import TabularLama, TabularLamaNN, TabularLamaUtilized
from .lightgbm import LightGBMClassification, LightGBMRegression
from .sklearn import (
    ExtraTreesClassification,
    ExtraTreesRegression,
    RandomForestClassification,
    RandomForestRegression,
    LogisticRegression,
    RidgeRegression,
)
from .xgboost import XGBClassification, XGBRegression

########################################################
# Each model is represented either as an instance name
# or a tuple, where:
#   1-st position is an instance name
#   2-nd position are additional parameters used on init
########################################################

linear_models = {
    "regression": [RidgeRegression],
    "classification": [LogisticRegression],
}

forest_models = {
    "regression": [RandomForestRegression, ExtraTreesRegression],
    "classification": [RandomForestClassification, ExtraTreesClassification],
}

boosting_models = {
    "regression": [
        CatBoostRegression, 
        XGBRegression, 
        LightGBMRegression,
    ],
    "classification": [
        CatBoostClassification,
        XGBClassification,
        LightGBMClassification,
    ],
}

lama_models = {
    "regression": [
        (TabularLama, {"task": "regression"}),
        (TabularLamaUtilized, {"task": "regression"}),
    ],
    "classification": [
        (TabularLama, {"task": "classification"}),
        (TabularLamaUtilized, {"task": "classification"}),
    ],
}

lama_nn_models = {
    "regression": [
        (TabularLamaNN, {"nn_name": "mlp", "task": "regression"}),
        (TabularLamaNN, {"nn_name": "denselight", "task": "regression"}),
        (TabularLamaNN, {"nn_name": "dense", "task": "regression"}),
        (TabularLamaNN, {"nn_name": "node", "task": "regression"}),
        (TabularLamaNN, {"nn_name": "autoint", "task": "regression"}),
        (TabularLamaNN, {"nn_name": "fttransformer", "task": "regression"}),
    ],
    "classification": [
        (TabularLamaNN, {"nn_name": "mlp", "task": "classification"}),
        (TabularLamaNN, {"nn_name": "denselight", "task": "classification"}),
        (TabularLamaNN, {"nn_name": "dense", "task": "classification"}),
        (TabularLamaNN, {"nn_name": "node", "task": "classification"}),
        (TabularLamaNN, {"nn_name": "autoint", "task": "classification"}),
        (TabularLamaNN, {"nn_name": "fttransformer", "task": "classification"}),
    ],
}

all_models = {
    "regression": linear_models["regression"]
    + forest_models["regression"]
    + boosting_models["regression"]
    + lama_models["regression"]
    + lama_nn_models["regression"],
    "classification": linear_models["classification"]
    + forest_models["classification"]
    + boosting_models["classification"]
    + lama_models["classification"]
    + lama_nn_models["classification"],
}


NAMES_MODELS_MAPPING = {
    "linear": linear_models,
    "forests": forest_models,
    "boostings": boosting_models,
    "lama": lama_models,
    "lama_nn": lama_nn_models,
    "all": all_models,
    "catboost": {
        "regression": [CatBoostRegression],
        "classification": [CatBoostClassification,],
    },
    "lightgbm": {
        "regression": [LightGBMRegression],
        "classification": [LightGBMClassification,],
    },
    "xgboost": {
        "regression": [XGBRegression],
        "classification": [XGBClassification,],
    },
    "test": {
        "regression": [LightGBMClassification, CatBoostRegression],
        "classification": [XGBClassification,
                        #    LightGBMClassification, 
                           CatBoostClassification,
                           ],
    }
}
