from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier, Pool
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from multiprocessing import cpu_count
import numpy as np
from automl.feature_selection.CustomMetrics import regression_roc_auc_score
from sklearn.metrics import mean_absolute_error

class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, task_type, target_colname, metric_name, metric_direction, timeout=120, random_state=42,
                 model='lama', strategy='RFA'):
        self.task_type = task_type
        self.target_colname = target_colname
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.timeout = timeout
        self.random_state = random_state
        self.model = model
        self.strategy = strategy

        assert self.metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"

    def calc_metric(self, train_with_prediction, test_with_prediction):
        if self.metric_name == 'mae':
            metric_test = mean_absolute_error(y_true=test_with_prediction[self.target_colname], y_pred=test_with_prediction[f'{self.target_colname}_prediction_{self.model}'])
            metric_train = mean_absolute_error(y_true=train_with_prediction[self.target_colname], y_pred=train_with_prediction[f'{self.target_colname}_prediction_{self.model}'])
        if self.metric_name == 'regression_roc_auc_score':
            metric_test = np.round(regression_roc_auc_score(y_true=test_with_prediction[self.target_colname].values, y_pred=test_with_prediction[f'{self.target_colname}_prediction_{self.model}'].values), 4)
            metric_train = np.round(regression_roc_auc_score(y_true=train_with_prediction[self.target_colname].values, y_pred=train_with_prediction[f'{self.target_colname}_prediction_{self.model}'].values), 4)
        
        return metric_train, metric_test

    def lama_fit_predict(self, train, test):

        cv_param = 4 if train.shape[0] // 5000 > 3 else 3
        automl = TabularAutoML(
                                task = Task(self.task_type),
                                timeout = self.timeout,
                                cpu_limit = cpu_count() - 1,
                                reader_params = {
                                                    'n_jobs': cpu_count() - 1, 
                                                    'cv': cv_param,
                                                    'random_state': self.random_state,
                                                    'advanced_roles': False,
                                                }
                            )
        oof_pred = automl.fit_predict(train, roles = {'target':self.target_colname}, verbose=-1)
        oof_pred = oof_pred.data
        prediction_train = automl.predict(train).data[:, 0]
        prediction_test = automl.predict(test).data[:, 0]
        train[f'{self.target_colname}_prediction_{self.model}'] = prediction_train
        test[f'{self.target_colname}_prediction_{self.model}'] = prediction_test
        lama_fi = automl.get_feature_scores()
        return lama_fi
    
    def RecursiveFeatureAddition(self, train, test):

        if self.model == 'lama':
            fi_first = self.lama_fit_predict(train, test)
        metric_train, metric_test = self.calc_metric(train, test)
        cnt_iters_from_last_best_metric_up = 0
        best_features_all = fi_first.sort_values('Importance', ascending=False).iloc[:]['Feature']
        best_features_iter = [best_features_all[0]]
        drop_features_iter = []
        iter_lst = [0]
        best_features_dict = {0:best_features_all}
        drop_features_dict = {}
        test_metric_lst = {0:metric_test}
        train_metric_lst = {0:metric_train}
        best_test_metric = 0
        metric_diff_train = {0:0}
        metric_diff_test = {0:0}
        print('Отобранные фичи:', best_features_iter)
        print('Метрика на всех фичах:', metric_test)
        for i in range(1, len(best_features_all)):
            iter_lst.append(i)
            feature = best_features_all[i]
            if self.model == 'lama':
                lama_fi = self.lama_fit_predict(train[best_features_iter + [feature] + [self.target_colname]], 
                                                test[best_features_iter + [feature] + [self.target_colname]])
            metric_train, metric_test = self.calc_metric(train, test)
            test_metric_lst[i] = metric_test
            train_metric_lst[i] = metric_train
            metric_diff_train[i] = train_metric_lst[i] - train_metric_lst[i-1]
            metric_diff_test[i] = test_metric_lst[i] - test_metric_lst[i-1]

            if self.metric_direction == 'maximize':
                if metric_test > best_test_metric:
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    best_features_iter.append(feature)
                    print('Отобранные фичи:', best_features_iter)
                    print('Лучшая метрика:', best_test_metric) 
                else:
                    drop_features_iter.append(feature)
                    cnt_iters_from_last_best_metric_up += 1
            elif self.metric_direction == 'minimize':
                if metric_test < best_test_metric:
                    print('Лучшая метрика:', best_test_metric)
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    best_features_iter.append(feature) 
                    print('Отобранные фичи:', best_features_iter)
                    print('Лучшая метрика:', best_test_metric) 
                else:
                    cnt_iters_from_last_best_metric_up += 1
                    drop_features_iter.append(feature)
            else:
                assert self.metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"
            best_features_dict[i] = best_features_iter
            drop_features_dict[i] = drop_features_iter
            # if early_stopper and cnt_iters_from_last_best_metric_up > early_stopper:
            #     print('early_stopper')
            #     break
        print(f'Отобрано {len(best_features_iter)} признаков: {best_features_iter}')    
        feature_selection_dict = {'iter_lst':iter_lst, 'train_metric_lst':train_metric_lst, 
                                'test_metric_lst':test_metric_lst, 'features':best_features_dict,
                                'metric_diff_train':metric_diff_train, 'metric_diff_test':metric_diff_test}
        #joblib.dump(feature_selection_dict, feature_selection_dict_path)
        
        return feature_selection_dict
    
    def fit(self, train, test):
        if self.strategy == 'RFA':
            feature_selection_dict = self.RecursiveFeatureAddition(train, test)
        return self
    def transform(self, X):
        return X