from automl.model.catboost import CatBoostClassification, CatBoostRegression
from automl.model.lightgbm import LightGBMClassification, LightGBMRegression
from automl.model.xgboost import XGBClassification, XGBRegression
from automl.model.sklearn import ExtraTreesClassification, ExtraTreesRegression
from automl.model.sklearn import RandomForestClassification, RandomForestRegression
from automl.model.sklearn import LogisticRegression, RidgeRegression
from automl.model.lama import TabularLamaClassification, TabularLamaRegression
from automl.model.lama import TabularLamaUtilizedClassification, TabularLamaUtilizedRegression


model_classes = [
    CatBoostClassification, CatBoostRegression,
    LightGBMClassification, LightGBMRegression,
    XGBClassification,  XGBRegression,
    ExtraTreesClassification, ExtraTreesRegression,
    RandomForestClassification, RandomForestRegression,
    LogisticRegression, RidgeRegression,
    ]

automl_model_classes = [
    TabularLamaClassification, TabularLamaRegression,
    TabularLamaUtilizedClassification, TabularLamaUtilizedRegression,
    ]

all_model_classes = model_classes + automl_model_classes