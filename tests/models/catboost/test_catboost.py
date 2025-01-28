# FILE: src/automl/model/catboost/test_catboost.py
import pytest
import inspect
import numpy as np
from automl.model.catboost import CatBoostClassification, CatBoostRegression
from catboost import CatBoostClassifier as CBClass, CatBoostRegressor as CBReg

from ..test_model import sample_data, sample_unbalanced_data
from unittest.mock import patch, MagicMock


def test_classification_init():
    model = CatBoostClassification()
    assert model.model_type == 'classification'
    assert model.name == "CatBoostClassification"
    assert model.model == CBClass
    assert model.model_predict_func_name == 'predict_proba'
    assert model.n_classes is None

def test_regression_init():
    model = CatBoostRegression()
    assert model.model_type == 'regression'
    assert model.name == "CatBoostRegression"
    assert model.model == CBReg
    assert model.model_predict_func_name == 'predict'
    assert model.n_classes is None

@pytest.mark.parametrize("model_class", [CatBoostRegression, CatBoostClassification, ])
def test_attributes(model_class):
    # Добавляем правильные параметры
    model = model_class(iterations=10, od_type='IncToDec', od_pval=0, eta=0.0005)
    assert model.eta == 0.0005
    assert model.od_type == 'IncToDec'

@pytest.mark.parametrize("model_class", [CatBoostRegression, CatBoostClassification, ])
def test_catboost_properties(model_class):
    model = model_class(iterations=10)
    CBClass(**model.inner_params)
    CBReg(**model.inner_params)

        
        