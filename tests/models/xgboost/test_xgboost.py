# FILE: src/automl/model/xgboost/xgboost.py
import pytest
import numpy as np
import xgboost as xgb

from automl.model.xgboost import XGBClassification, XGBRegression

from ..test_model import sample_data, sample_unbalanced_data


def test_xgboost_classification_init():
    model = XGBClassification()
    assert model.model_type == 'classification'
    assert model.name == "XGBClassification"
    assert model.objective == 'binary:logistic'
    assert model.num_class is None

def test_xgboost_regression_init():
    model = XGBRegression()
    assert model.model_type == 'regression'
    assert model.name == "XGBRegression"
    assert model.objective == 'reg:squarederror'

@pytest.mark.parametrize("model_class", [XGBClassification, XGBRegression])
def test_attributes(model_class):
    model = model_class(iterations=10, eta=0.055, class_weight='balanced')
    assert model.eta == 0.055

@pytest.mark.parametrize("model_class", [XGBClassification, XGBRegression])
def test_inner_properties(sample_data, model_class):
    X, y = sample_data
    iterations = 10
    model = model_class(iterations=iterations, early_stopping_round=-1)
    dtrain = xgb.DMatrix(X, label=y, silent=True, enable_categorical=True,)
    model = xgb.train(params=model.inner_params, dtrain=dtrain, num_boost_round=iterations)
    
def test_xgboost_classification_class_weight(sample_unbalanced_data):
    from sklearn.metrics import roc_auc_score
    
    X, y = sample_unbalanced_data
    model = XGBClassification(class_weight='balanced', num_boost_round=2)
    model.fit(X, y)
    
    assert "scale_pos_weight" in model.inner_params
    assert model.inner_params.get('scale_pos_weight', None) is not None

    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    
    score = roc_auc_score(y, preds[:, 1])

    # Fit another model without class_weight for comparison
    model_no_weight = XGBClassification(class_weight=None, num_boost_round=2)
    assert "scale_pos_weight" not in model_no_weight.inner_params or model_no_weight.inner_params.get('scale_pos_weight', None) is None
    model_no_weight.fit(X, y)
    preds_no_weight = model_no_weight.predict(X)
    score_no_weight = roc_auc_score(y, preds_no_weight[:, 1])

    # Check if accuracies are different
    assert not (score_no_weight == pytest.approx(score, rel=1e-3))
    
# def test_custom_metric(sample_data):
#     from sklearn.metrics import roc_auc_score
#     from automl.metrics import get_scorer
#     from automl.model.lightgbm.metrics import get_custom_lightgbm_metric
    
#     scorer = get_scorer('roc_auc')
#     custom_metric = get_custom_lightgbm_metric(scorer.score, scorer.greater_is_better)
    
#     X, y_true = sample_data
#     y_pred = np.random.rand(len(y_true))
#     data = lgb.Dataset(data=X, label=y_true)
    
#     metric_name, metric_value, greater_is_better = custom_metric(y_pred, data)
#     assert metric_name == 'custom_metric'
#     assert greater_is_better
#     assert metric_value == roc_auc_score(y_true, y_pred)
    
# def test_custom_metric_in_eval_metric(sample_data):
#     from automl.metrics import get_scorer
#     from automl.model.lightgbm.metrics import get_custom_lightgbm_metric
    
#     scorer = get_scorer('roc_auc')
#     custom_metric = get_custom_lightgbm_metric(scorer.score, scorer.greater_is_better)
    
#     X, y = sample_data
    
#     model = LightGBMClassification(
#         is_unbalance='true', 
#         num_iteration=2, 
#         eval_metric=custom_metric, 
#         n_splits=2, 
#         num_leaves=2,)
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     custom_score = scorer.score(y, y_pred[:, 1])
    
#     model = LightGBMClassification(
#         is_unbalance='true', 
#         num_iteration=2, 
#         eval_metric='auc', 
#         n_splits=2, 
#         num_leaves=2,)
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     score = scorer.score(y, y_pred[:, 1])
    
#     assert custom_score == pytest.approx(score, rel=1e-3)

@pytest.mark.parametrize("model_class", [XGBClassification, XGBRegression])
def test_early_stopping(sample_data, model_class):
    X, y = sample_data
    num_boost_round = 100
    n_splits = 2
    early_stopping_rounds_1 = 10
    early_stopping_rounds_2 = 1

    model1 = model_class(
        num_boost_round=num_boost_round, 
        early_stopping_rounds=early_stopping_rounds_1,
        eta=0.1, max_depth=2,
        n_splits=n_splits,
    )
    model1.tune(X, y)
    model1.fit(X, y)
    
    
    model2 = model_class(
        num_boost_round=num_boost_round, 
        early_stopping_rounds=early_stopping_rounds_2,
        eta=0.1, max_depth=2,
        n_splits=n_splits,
    )
    model2.tune(X, y)
    model2.fit(X, y)
    
    for i in range(n_splits):
        iteration1 = model1.models[i].best_iteration
        iteration2 = model2.models[i].best_iteration
        assert iteration1 < num_boost_round
        assert iteration2 < num_boost_round
        assert iteration2 <= iteration1
        
    iteration1 = model1.best_params['num_boost_round']
    iteration2 = model2.best_params['num_boost_round']
    assert iteration1 < num_boost_round
    assert iteration2 < num_boost_round
    assert iteration2 <= iteration1
