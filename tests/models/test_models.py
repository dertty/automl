# FILE: src/automl/model/catboost/test_catboost.py
import pytest
import inspect
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from optuna import create_study
from unittest.mock import patch, MagicMock

from .fixtures import sample_data, sample_unbalanced_data
from .models_lists import model_classes, all_model_classes, automl_model_classes
from .utils import check_n_classes


default_model_params = {
    'num_iterations': 1, 
    'n_splits': 2, 
    'n_jobs': 1, 
    'device_type': 'cpu', 
    'random_state': 42,
    'verbose': 0
    }


@pytest.mark.parametrize("model_class", all_model_classes)
def test_common_attributes(sample_data, model_class):
    model = model_class()
    
    # Проверяем, что параметры, не относящиеся к "оригинальной" модели не попадают в inner_params
    inner_params = model.inner_params
    assert 'name' not in inner_params
    assert 'num_class' not in inner_params
    assert 'max_iterations' not in inner_params
    assert 'categorical_feature' not in inner_params
    assert 'categorical_features' not in inner_params
    assert 'oof_preds' not in inner_params
    assert 'best_params' not in inner_params
    assert 'time_series' not in inner_params
    assert 'n_splits' not in inner_params
    assert 'n_folds' not in inner_params
    assert '_not_inner_model_params' not in inner_params
    assert 'model_type' not in inner_params
    assert 'model' not in inner_params 
    assert 'models' not in inner_params
    assert 'model_predict_func_name' not in inner_params
    
    # У всех моделей должны быть следующие методы
    assert 'fit' in dir(model)
    assert 'predict' in dir(model)
    assert 'tune' in dir(model)
    
    # Проверяем, что параметры по умолчанию задаются правильно
    assert model.categorical_feature == None
    assert model.best_params == {}
    assert isinstance(model.random_state, int)
    assert model.random_state >= 0
    assert model.device_type.lower() == 'cpu'
    assert model.n_jobs >= -1
    
    X, y = sample_data
    n_jobs = 1
    random_state = 42
    model = model_class(num_iterations=2, n_splits=2, n_jobs=n_jobs, random_state=random_state)
    # Проверяем, что в процеесе выполнения методов не меняются параметры
    # А так же, что всегджа рассчитвается правильное количество классов
    def check_that_params_doesnot_change(model, random_state, n_jobs):
        assert model.categorical_feature == []
        assert type(model.best_params) == dict
        assert model.random_state == random_state
        assert model.device_type.lower() == 'cpu'
        assert model.n_jobs == n_jobs
    
    # проверяем fit 
    oof_preds = model.fit(X, y,)
    check_that_params_doesnot_change(model, random_state, n_jobs)
    check_n_classes(model, oof_preds, y)
    assert oof_preds.shape[0] == X.shape[0]
    
    preds = model.predict(X)
    check_that_params_doesnot_change(model, random_state, n_jobs)
    check_n_classes(model, preds, y)
    assert preds.shape[0] == X.shape[0]
    
    # проверяем tune 
    model.tune(X, y,)
    check_that_params_doesnot_change(model, random_state, n_jobs)
    check_n_classes(model, oof_preds, y)
    
    oof_preds = model.fit(X, y,)
    check_that_params_doesnot_change(model, random_state, n_jobs)
    check_n_classes(model, oof_preds, y)
    assert oof_preds.shape[0] == X.shape[0]

    preds = model.predict(X)
    check_that_params_doesnot_change(model, random_state, n_jobs)
    check_n_classes(model, preds, y)
    assert preds.shape[0] == X.shape[0]
    
# Для все моделей, кроме тех, у которых встроенный тюнинг должен быть best_params
@pytest.mark.parametrize("model_class", model_classes)
def test_best_params_attribute_for_base_models(sample_data, model_class):
    X, y = sample_data
    model = model_class(**default_model_params)
    model.tune(X, y,)
    assert len(model.best_params.keys()) > 0
    
# Для automl best_params остаётся пустой
@pytest.mark.parametrize("model_class", automl_model_classes)
def test_best_params_attribute_for_automl_models(sample_data, model_class):
    X, y = sample_data
    model = model_class(**default_model_params)
    model.tune(X, y,)
    assert model.best_params == {}

# Если в предикт передаётся список таблиц, то возвращается список предиктов
@pytest.mark.parametrize("model_class", all_model_classes)
def test_list_predict(sample_data, model_class):
    X, y = sample_data
    model = model_class(**default_model_params)
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
def test_np_categorical_features(sample_data, model_class):
    model = model_class(**default_model_params)
    assert model.categorical_feature == None
    
    X, y = sample_data
    indexed_categorical_feature = [0, 1]
    categorical_feature=['column_0', 'column_1']
    
    oof_preds = model.fit(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    assert oof_preds.shape[0] == X.shape[0]
    
    model.tune(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    oof_preds = model.fit(X, y, categorical_feature=indexed_categorical_feature)
    assert model.categorical_feature == categorical_feature
    assert oof_preds.shape[0] == X.shape[0]
    
@pytest.mark.parametrize("model_class", all_model_classes)
def test_pd_categorical_features(sample_data, model_class):
    from automl.model.utils import convert_to_pandas
    
    model = model_class(**default_model_params)
    assert model.categorical_feature == None
    
    X, y = sample_data
    X = convert_to_pandas(X)
    categorical_feature=['column_0', 'column_1']
    
    oof_preds = model.fit(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature
    assert oof_preds.shape[0] == X.shape[0]
    
    model.tune(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature
    
    oof_preds = model.fit(X, y, categorical_feature=categorical_feature)
    assert model.categorical_feature == categorical_feature
    assert oof_preds.shape[0] == X.shape[0]

@pytest.mark.parametrize("model_class", model_classes)
def test_prepare_data(sample_data, model_class):    
    model = model_class(**default_model_params)
    X, y = sample_data
    
    X_prepared = model._prepare(X)
    assert X_prepared.shape == X.shape
    assert model.categorical_feature == []
    
    X_prepared, y_prepared = model._prepare(X, y)
    assert X_prepared.shape == X.shape
    assert y_prepared.shape == y.shape
    assert model.categorical_feature == []
    
    model._prepare(X, categorical_feature=[])
    assert model.categorical_feature == []
    model._prepare(X, categorical_feature=None)
    assert model.categorical_feature == []
    model._prepare(X, categorical_feature=[1, 2])
    assert model.categorical_feature == ['column_1', 'column_2']
    model._prepare(pd.DataFrame(X, columns=[f'a{i}' for i in range(X.shape[1])]), categorical_feature=['a1',])
    assert model.categorical_feature == ['a1']

@pytest.mark.parametrize("model_class", model_classes)
def test_get_base_trial_params(model_class):
    study = create_study()
    trial = study.ask()
    params = model_class.get_base_trial_params(trial)
    assert isinstance(params, dict)
    assert len(params) > 0
    params = model_class.get_trial_params(trial)
    assert isinstance(params, dict)
    assert len(params) > 0

@pytest.mark.parametrize("model_class", model_classes)
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

@pytest.mark.parametrize("model_class", all_model_classes)
@pytest.mark.parametrize("model_type, eval_metric, metrics_dict, expected", [
    ('classification', 'accuracy', {'accuracy': True}, True),
    ('classification', 'log_loss', {'log_loss': False}, False),
    ('regression', 'mse', {'mse': False}, False),
    ('regression', 'r2', {'r2': True}, True),
    ('classification', 'unknown_metric', {}, True),
    ('regression', 'unknown_metric', {}, False),
])
def test_get_str_greater_is_better(model_class, model_type, eval_metric, metrics_dict, expected):
    result = model_class.get_str_greater_is_better(model_type, eval_metric, metrics_dict)
    assert result == expected

@pytest.mark.parametrize("model_class", all_model_classes)
@pytest.mark.parametrize("X, y, train_idx, test_idx, expected_X_train, expected_y_train, expected_X_test, expected_y_test", [
    (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]), [0, 1], [2, 3], 
     np.array([[1, 2], [3, 4]]), np.array([1, 2]), np.array([[5, 6], [7, 8]]), np.array([3, 4])),
    (pd.DataFrame({'a': [1, 3, 5, 7], 'b': [2, 4, 6, 8]}), np.array([1, 2, 3, 4]), [0, 1], [2, 3],
     pd.DataFrame({'a': [1, 3], 'b': [2, 4]}), np.array([1, 2]), pd.DataFrame({'a': [5, 7], 'b': [6, 8]}), np.array([3, 4])),
])
def test_get_train_test_data(model_class, X, y, train_idx, test_idx, expected_X_train, expected_y_train, expected_X_test, expected_y_test):
    X_train, y_train, X_test, y_test = model_class._get_train_test_data(X, y, train_idx, test_idx)
    pd.testing.assert_frame_equal(pd.DataFrame(X_train).reset_index(drop=True), pd.DataFrame(expected_X_train).reset_index(drop=True))
    np.testing.assert_array_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(pd.DataFrame(X_test).reset_index(drop=True), pd.DataFrame(expected_X_test).reset_index(drop=True))
    np.testing.assert_array_equal(y_test, expected_y_test)