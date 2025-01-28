# FILE: src/automl/model/catboost/test_catboost.py
import pytest
import inspect
import numpy as np

from optuna import create_study
from unittest.mock import patch, MagicMock

from automl.model.catboost import CatBoostClassification, CatBoostRegression
from automl.model.lightgbm import LightGBMClassification, LightGBMRegression
from automl.model.xgboost import XGBClassification, XGBRegression
from automl.model.sklearn import ExtraTreesClassification, ExtraTreesRegression
from automl.model.sklearn import RandomForestClassification, RandomForestRegression
from automl.model.sklearn import LogisticRegression, RidgeRegression


all_model_classes = [
    CatBoostClassification, CatBoostRegression,
    LightGBMClassification, LightGBMRegression,
    XGBClassification,  XGBRegression,
    ExtraTreesClassification, ExtraTreesRegression,
    RandomForestClassification, RandomForestRegression,
    LogisticRegression, RidgeRegression,
    ]

@pytest.fixture
def sample_data():
    n_rows = 50
    n_cols = 5
    X = np.random.rand(n_rows, n_cols)
    y = (X.sum(axis=1) + np.random.normal(n_cols / 2, n_cols / 2, X.shape[0]) > n_cols).astype(int)
    return X, y

@pytest.fixture
def sample_unbalanced_data():
    X = np.random.rand(100, 10)
    y = np.concatenate([np.zeros(95), np.ones(5)])
    return X, y


@pytest.mark.parametrize("model_class", all_model_classes)
def test_common_attributes(sample_data, model_class):
    model = model_class()
    inner_params = model.inner_params
    
    assert 'name' not in inner_params
    assert 'num_class' not in inner_params
    assert 'max_iterations' not in inner_params
    assert 'categorical_feature' not in inner_params
    assert 'models' not in inner_params
    assert 'oof_preds' not in inner_params
    assert 'best_params' not in inner_params
    assert 'time_series' not in inner_params
    assert 'n_splits' not in inner_params
    assert '_not_inner_model_params' not in inner_params
    assert 'model_type' not in inner_params
    assert 'model' not in inner_params
    assert 'model_predict_func_name' not in inner_params
    
    assert 'fit' in dir(model)
    assert 'predict' in dir(model)
    assert 'tune' in dir(model)
    
    assert model.categorical_feature == None
    assert model.best_params == {}
    assert isinstance(model.random_state, int)
    assert model.device_type.lower() == 'cpu'
    assert model.n_jobs > 0
    
    X, y = sample_data
    model = model_class(n_splits=2, n_jobs=1, num_iterations=2, random_state=42)
    model.fit(X, y,)
    
    assert model.categorical_feature == []
    assert model.best_params == {}
    assert model.random_state == 42
    assert model.device_type.lower() == 'cpu'
    assert model.n_jobs == 1
    
    model.tune(X, y,)
    
    assert model.categorical_feature == []
    assert len(model.best_params.keys()) > 0
    assert model.random_state == 42
    assert model.device_type.lower() == 'cpu'
    assert model.n_jobs == 1
    
    oof_preds = model.fit(X, y,)
    assert model.categorical_feature == []
    assert len(model.best_params.keys()) > 0
    assert model.random_state == 42
    assert model.device_type.lower() == 'cpu'
    assert model.n_jobs == 1
    
    n_classes = model.__dict__.get('num_class', model.__dict__.get('n_classes', None))
    if n_classes:
        assert oof_preds.shape[1] == n_classes
    else:
        assert oof_preds.ndim == 1

@pytest.mark.parametrize("model_class", all_model_classes)
def test_np_categorical_features(sample_data, model_class):
    model = model_class(n_splits=2, n_jobs=1, num_iterations=2)
    assert model.categorical_feature == None
    
    X, y = sample_data
    indexed_categorical_feature = [0, 1]
    categorical_feature=['column_0', 'column_1']
    
    oof_preds = model.fit(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    model.tune(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    oof_preds = model.fit(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    
@pytest.mark.parametrize("model_class", all_model_classes)
def test_pd_categorical_features(sample_data, model_class):
    from automl.model.utils import convert_to_pandas
    
    model = model_class(n_splits=2, n_jobs=1, num_iterations=2)
    assert model.categorical_feature == None
    
    X, y = sample_data
    X = convert_to_pandas(X)
    categorical_feature=['column_0', 'column_1']
    
    oof_preds = model.fit(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    model.tune(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    oof_preds = model.fit(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature

@pytest.mark.parametrize("model_class", all_model_classes)
def test_predict(sample_data, model_class):
    X, y = sample_data
    model = model_class(verbose=0, n_splits=2, n_jobs=1, num_iterations=2, random_state=0, device_type='cpu')
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == y.shape[0]
    n_classes = model.__dict__.get('num_class', model.__dict__.get('n_classes', None))
    if n_classes:
        assert preds.shape[1] == n_classes
    else:
        assert preds.ndim == 1
        
@pytest.mark.parametrize("model_class", all_model_classes)
def test_list_predict(sample_data, model_class):
    X, y = sample_data
    model = model_class(verbose=0, n_splits=2, n_jobs=1, num_iterations=2, random_state=0, device_type='cpu')
    model.fit(X, y)
    Xs = [X, X[:10, :], X]
    preds = model.predict(Xs)
    
    assert isinstance(preds, list)
    assert len(preds) == 3
    if model.model_type == 'classification':
        for pred, x in zip(preds, Xs):
            assert pred.shape == (x.shape[0], 2)
    elif model.model_type == 'regression':
        for pred, x in zip(preds, Xs):
            assert pred.shape == (x.shape[0],)


@pytest.mark.parametrize("model_class", all_model_classes)
def test_prepare_data(sample_data, model_class):
    model = model_class()
    X, y = sample_data
    X_prepared, y_prepared = model._prepare(X, y)
    assert X_prepared.shape == X.shape
    assert y_prepared.shape == y.shape


@pytest.mark.parametrize("model_class", all_model_classes)
def test_get_base_trial_params(model_class):
    study = create_study()
    trial = study.ask()
    params = model_class.get_base_trial_params(trial)
    params = model_class.get_trial_params(trial)


@pytest.mark.parametrize("model_class", all_model_classes)
def test_tune_with_custom_study(sample_data, model_class):
    X, y = sample_data
    model = model_class(num_iterations=2)
    
    custom_study = MagicMock()
    custom_study.best_params = {'depth': 4, 'learning_rate': 0.05}
    custom_study.best_trial.user_attrs = {"iterations": 50, 'num_boost_round': 50, 'num_iterations': 50}
    custom_study.trials = [MagicMock() for _ in range(5)]
    
    with patch('optuna.create_study', return_value=custom_study):
        model.tune(X, y, timeout=10)
        assert model.best_params['depth'] == 4
        assert model.best_params['learning_rate'] == 0.05
        assert model.inner_params['depth'] == 4
        assert model.inner_params['learning_rate'] == 0.05