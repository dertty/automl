# FILE: src/automl/model/lightgbm/test_lightgbm.py
import pytest
import numpy as np
import lightgbm as lgb
from optuna import create_study

from automl.model.lightgbm import LightGBMClassification, LightGBMRegression
from automl.model.lightgbm.lightgbm import LightGBMBase


from ..test_models import sample_data, sample_unbalanced_data


def test_classification_init():
    model = LightGBMClassification()
    assert model.model_type == 'classification'
    assert model.name == "LightGBMClassification"
    assert model.objective == 'binary'
    assert model.num_class is None

def test_regression_init():
    model = LightGBMRegression()
    assert model.model_type == 'regression'
    assert model.name == "LightGBMRegression"
    assert model.objective == 'regression'

@pytest.mark.parametrize("model_class", [LightGBMClassification, LightGBMRegression])
def test_not_tuned_params(model_class):
    model = model_class()
    not_tuned_params = model.not_tuned_params
    inner_params = model.inner_params
    
    assert "num_threads" in not_tuned_params
    assert "seed" in not_tuned_params
    assert "verbosity" in not_tuned_params
    assert "device_type" in not_tuned_params
    assert "early_stopping_round" in not_tuned_params
    assert "early_stopping_min_delta" in not_tuned_params
    assert "objective" in not_tuned_params
    assert "metric" in not_tuned_params
    
    assert "num_threads" in inner_params
    assert "seed" in inner_params
    assert "verbosity" in inner_params
    assert "device_type" in inner_params
    assert "early_stopping_round" in inner_params
    assert "early_stopping_min_delta" in inner_params
    assert "objective" in inner_params
    assert "metric" in inner_params
    

@pytest.mark.parametrize("model_class", [LightGBMClassification, LightGBMRegression])
def test_attributes(sample_data, model_class):
    # Проверяем, что можно добавить любой параметр для модели
    n_splits = 2
    num_leaves = 2
    n_estimators = 7
    boosting_type = 'rf'
    model = model_class(
        n_splits=n_splits,
        iterations=n_estimators, 
        boosting_type=boosting_type, 
        num_leaves=num_leaves,)
    inner_params = model.inner_params
    
    assert model.boosting_type == 'rf'
    assert "boosting_type" in inner_params
    assert inner_params["boosting_type"] == boosting_type
    
    assert model.num_iterations == n_estimators
    assert "num_iterations" in inner_params
    assert inner_params["num_iterations"] == n_estimators
    
    assert model.num_leaves == num_leaves
    assert "num_leaves" in inner_params
    assert inner_params["num_leaves"] == num_leaves


@pytest.mark.parametrize("model_class", [LightGBMClassification, LightGBMRegression])
def test_inner_properties(sample_data, model_class):
    # проверяем, что inner_params правильный
    # Если добавляем для "одинаковых" параметра, то используется только один для lgb
    X, y = sample_data
    n_estimators = 10
    num_leaves = 2
    n_splits = 2
    model = model_class(
        n_estimators=n_estimators, 
        n_splits=n_splits, 
        objective='binary', 
        objective_type='multiclass', 
        early_stopping_round=-1, 
        num_leaves=num_leaves,)
    model.fit(X, y)
    for i in range(n_splits):
        params = model.models[i].params
        assert params['objective'] == 'binary'
        assert 'objective_type' not in params
    
    model = lgb.train(params=model.inner_params, train_set=lgb.Dataset(X, y), num_boost_round=1_000)
    assert model.current_iteration() == n_estimators


def test_classification_class_weight(sample_unbalanced_data):
    from sklearn.metrics import roc_auc_score
    
    X, y = sample_unbalanced_data
    params = {'num_iteration': 2, 'num_leaves': 2,}
    model = LightGBMClassification(is_unbalance='true', **params)
    model.fit(X, y)
    assert "is_unbalance" in model.inner_params
    assert model.inner_params['is_unbalance'] == 'true'
   
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    score = roc_auc_score(y, preds[:, 1])

    # Fit another model without class_weight for comparison
    model_no_weight = LightGBMClassification(is_unbalance='false', **params)
    model_no_weight.fit(X, y)
    assert model_no_weight.inner_params['is_unbalance'] == 'false'
    preds_no_weight = model_no_weight.predict(X)
    score_no_weight = roc_auc_score(y, preds_no_weight[:, 1])

    # Check if accuracies are different
    assert not (score_no_weight == pytest.approx(score, rel=1e-3))


def test_custom_metric(sample_data):
    from sklearn.metrics import roc_auc_score
    from automl.metrics import get_scorer
    from automl.model.lightgbm.metrics import get_custom_lightgbm_metric
    
    scorer = get_scorer('roc_auc')
    name = 'roc_auc_custom_metric'
    custom_metric = get_custom_lightgbm_metric(scorer.score, scorer.greater_is_better, name)
    
    X, y_true = sample_data
    y_pred = np.random.rand(len(y_true))
    data = lgb.Dataset(data=X, label=y_true)
    
    metric_name, metric_value, greater_is_better = custom_metric(y_pred, data)
    assert metric_name == name
    assert greater_is_better
    assert metric_value == roc_auc_score(y_true, y_pred)


def test_custom_metric_in_eval_metric(sample_data):
    from automl.metrics import get_scorer
    from automl.model.lightgbm.metrics import get_custom_lightgbm_metric
    
    scorer = get_scorer('roc_auc')
    custom_metric = get_custom_lightgbm_metric(scorer.score, scorer.greater_is_better)
    
    X, y = sample_data
    params = {
        'is_unbalance': 'true', 
        'num_iteration': 2, 
        'n_splits': 2, 
        'num_leaves': 2,
    }
    
    model = LightGBMClassification(eval_metric=custom_metric, **params)
    model.fit(X, y)
    y_pred = model.predict(X)
    custom_score = scorer.score(y, y_pred[:, 1])
    
    model = LightGBMClassification(eval_metric='auc', **params)
    model.fit(X, y)
    y_pred = model.predict(X)
    score = scorer.score(y, y_pred[:, 1])
    
    assert custom_score == pytest.approx(score)


@pytest.mark.parametrize("model_class", [LightGBMClassification, LightGBMRegression])
def test_early_stopping(sample_data, model_class):
    X, y = sample_data
    params = {
        'num_iteration': 100,
        'n_splits': 2,
        'num_leaves': 2,
    }
    
    model1 = model_class(early_stopping_round=10, **params)
    model1.tune(X, y)
    model1.fit(X, y)
    
    model2 = model_class(early_stopping_round=1, **params)
    model2.tune(X, y)
    model2.fit(X, y)
    
    for i in range(params['n_splits']):
        iteration1 = model1.models[i].current_iteration()
        iteration2 = model2.models[i].current_iteration()
        assert iteration1 < params['num_iteration']
        assert iteration2 < params['num_iteration']
        assert iteration2 <= iteration1
        
    iteration1 = model1.best_params['num_iterations']
    iteration2 = model2.best_params['num_iterations']
    assert iteration1 < params['num_iteration']
    assert iteration2 < params['num_iteration']
    assert iteration2 <= iteration1
    