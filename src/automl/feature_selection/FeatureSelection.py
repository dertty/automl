from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import joblib
from src.CustomMetrics import regression_roc_auc_score
from sklearn.metrics import mean_absolute_error
from multiprocessing import cpu_count
import numpy as np


def lama_fit_predict(train_, test_, task_type, random_state, target_colname, metric_name, timeout=700):
    '''Обучение и предикт Lama.
        Args:
           train_: обучающая выборка
           test_: тестовая выборка
           task_type: решаемая задача (регрессия = 'reg', бинарная классификация = 'binary', мультиклассовая классификация = 'multiclass')
           random_state: random_state
           target_colname: название столбца с целевой переменной
           metric_name: наименование метрики
           timeout: timeout для обучения Ламы
        Returns:
           metric_train: метрика на трейне
           metric_test: метрика на тесте 
           lama_fi: feature importance
    '''
    train = train_.copy()
    test = test_.copy()
    cv_param = 4 if train.shape[0] // 5000 > 3 else 3
    automl = TabularAutoML(
                            task = Task(task_type),
                            timeout = timeout,
                            cpu_limit = cpu_count() - 1,
                            reader_params = {
                                                'n_jobs': cpu_count() - 1, 
                                                'cv': cv_param,
                                                'random_state': random_state,
                                                'advanced_roles': False,
                                            }
                          )
    oof_pred = automl.fit_predict(train, roles = {'target':target_colname}, verbose=-1)
    oof_pred = oof_pred.data
    prediction_train = automl.predict(train).data[:, 0]
    prediction_test = automl.predict(test).data[:, 0]
    train[f'{target_colname}_prediction'] = prediction_train
    test[f'{target_colname}_prediction'] = prediction_test
    if metric_name == 'mae':
        metric_test = mean_absolute_error(y_true=test[target_colname], y_pred=test[f'{target_colname}_prediction'])
        metric_train = mean_absolute_error(y_true=train[target_colname], y_pred=train[f'{target_colname}_prediction'])
    if metric_name == 'regression_roc_auc_score':
        metric_test = np.round(regression_roc_auc_score(y_true=test[target_colname].values, y_pred=test[f'{target_colname}_prediction'].values), 4)
        metric_train = np.round(regression_roc_auc_score(y_true=train[target_colname].values, y_pred=train[f'{target_colname}_prediction'].values), 4)
    lama_fi = automl.get_feature_scores()
    return metric_train, metric_test, lama_fi


def lama_feature_selection(train_, test_, task_type, random_state, target_colname, metric_name, metric_direction, timeout=100, feature_selection_dict_path='outputs/feature_selection_dict.joblib', early_stopper = None):
    '''Recursive feature elimination с обучением модели LAMA.
        Args:
           train_: обучающая выборка
           test_: тестовая выборка
           task_type: решаемая задача (регрессия = 'reg', бинарная классификация = 'binary', мультиклассовая классификация = 'multiclass')
           random_state: random_state
           target_colname: название столбца с целевой переменной
           metric_name: наименование метрики
           metric_direction: направление улучшения метрики (maximize or minimize)
           timeout: timeout для обучения Ламы
        Returns:
           —Сохранем в lama_rfecv_results_file_name словарь lama_rfecv_dict, который содержит iter_list, train_metric_lst, test_metric_lst, drop_features
           None
    '''
    train = train_.copy()
    test = test_.copy()

    metric_train, metric_test, lama_fi_first = lama_fit_predict(train, test, task_type, random_state, target_colname, metric_name, timeout=timeout)
    cnt_iters_from_last_best_metric_up = 0
    best_features_all = lama_fi_first.sort_values('Importance', ascending=False).iloc[:]['Feature']
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
        metric_train, metric_test, lama_fi = lama_fit_predict(
                                                                train[best_features_iter + [feature] + [target_colname]], 
                                                                test[best_features_iter + [feature] + [target_colname]], 
                                                                task_type, random_state, target_colname, metric_name, 
                                                                timeout=timeout
                                                             )
        test_metric_lst[i] = metric_test
        train_metric_lst[i] = metric_train
        metric_diff_train[i] = train_metric_lst[i] - train_metric_lst[i-1]
        metric_diff_test[i] = test_metric_lst[i] - test_metric_lst[i-1]

        if metric_direction == 'maximize':
            if metric_test > best_test_metric:
                best_test_metric = metric_test
                cnt_iters_from_last_best_metric_up = 0
                best_features_iter.append(feature)
                print('Отобранные фичи:', best_features_iter)
                print('Лучшая метрика:', best_test_metric) 
            else:
                drop_features_iter.append(feature)
                cnt_iters_from_last_best_metric_up += 1
        elif metric_direction == 'minimize':
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
            assert metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"
        best_features_dict[i] = best_features_iter
        drop_features_dict[i] = drop_features_iter
        if early_stopper and cnt_iters_from_last_best_metric_up > early_stopper:
            print('early_stopper')
            break
    print(f'Отобрано {len(best_features_iter)} признаков: {best_features_iter}')    
    feature_selection_dict = {'iter_lst':iter_lst, 'train_metric_lst':train_metric_lst, 
                              'test_metric_lst':test_metric_lst, 'features':best_features_dict,
                              'metric_diff_train':metric_diff_train, 'metric_diff_test':metric_diff_test}
    joblib.dump(feature_selection_dict, feature_selection_dict_path)
    
    return feature_selection_dict
