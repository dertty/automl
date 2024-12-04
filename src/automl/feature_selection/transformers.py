import numbers
import pandas as pd
import numpy as np
from feature_engine.selection import SmartCorrelatedSelection, DropHighPSIFeatures
from sklearn.ensemble._bagging import _generate_indices
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils import Bunch, _safe_indexing, check_array, check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier, Pool
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from multiprocessing import cpu_count
#from automl.feature_selection.CustomMetrics import regression_roc_auc_score
from ..loggers import get_logger
from .utils import SmartCorrelatedSelectionFast


from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType
from catboost import Pool

log = get_logger(__name__)

class AdversarialTestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, split_col='is_test_for_val', random_state=42, auc_trshld = 0.65):
        '''Adversarial test.
            Args:
                df_train: обучающая выборка
                df_test: тестовая выборка
                random_state: random_state
                auc_trshld: граница для метрики
            Returns:
                ignore_features: признаки, не прошедшие adversarial тест
        '''
        self.split_col = split_col
        self.random_state = random_state
        self.auc_trshld = auc_trshld
        self.adversarial_drop_features = []
 
    def cb_feature_importance(self, model):

        '''Метод возвращает feature importance модели (Catboost)'''
        #Create arrays from feature importance and feature names
        feature_importance = model.feature_importances_
        feature_names = model.feature_names_
        #Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
       
        return fi_df
    
    def create_adversarial_data(self, df_master):

        n_val = df_master[df_master[self.split_col]==1].shape[0]
        adversarial_val = df_master.sample(n_val, replace=False)
        adversarial_train = df_master[~df_master.index.isin(adversarial_val.index)]
    
        return adversarial_train, adversarial_val

    def fit(self, X, y=None):

        target = self.split_col
        adversarial_train, adversarial_test = self.create_adversarial_data(X)
        train_data = Pool(data=adversarial_train.drop(target, axis=1), label=adversarial_train[target])
        holdout_data = Pool(
                            data=adversarial_test.drop(target, axis=1),
                            label=adversarial_test[target]
                        )
        ignore_features = []
        params = {
                'iterations': 10,
                'eval_metric': 'AUC',
                'od_type': 'Iter',
                'od_wait': 5,
                'random_seed': self.random_state,
                'ignored_features': [],
                'depth': 4,
                "verbose": False
            }
        model = CatBoostClassifier(**params)
        model.fit(train_data, eval_set=holdout_data)
        model_auc = model.best_score_['validation']['AUC']
        cb_feature_importance_df = self.cb_feature_importance(model)
        top_fi = cb_feature_importance_df.iloc[0]['feature_importance']
        top_fi_name = cb_feature_importance_df.iloc[0]['feature_names']
        while (model_auc > self.auc_trshld) & (top_fi != 0):
            ignore_features.append(top_fi_name)
            params.update({'ignored_features': ignore_features})
            model = CatBoostClassifier(**params)
            model.fit(train_data, eval_set=holdout_data)
            model_auc = model.best_score_['validation']['AUC']
            cb_feature_importance_df = self.cb_feature_importance(model)
            top_fi = cb_feature_importance_df.iloc[0]['feature_importance']
            top_fi_name = cb_feature_importance_df.iloc[0]['feature_names']
        self.adversarial_drop_features = ignore_features
        
        if len(self.adversarial_drop_features) > 0:
            log.info(f"Features not passing adversarial test to drop: {self.adversarial_drop_features}", msg_type="val_tests")
        
        return self
    
    def transform(self, X):
        X = X.drop(self.adversarial_drop_features, axis=1)
        
        return X
    
    
class DropHighPSITransformer(BaseEstimator, TransformerMixin):
    def __init__(self, split_col='is_test_for_val', psi_cut_off=0.5, psi_threshold=0.2,
                 psi_bins=15, psi_strategy='equal_width', psi_missing_values='ignore'):
        '''PSI test.
        '''
        self.split_col = split_col
        self.psi_cut_off = psi_cut_off
        self.psi_threshold = psi_threshold
        self.psi_bins = psi_bins
        self.psi_strategy = psi_strategy
        self.psi_missing_values = psi_missing_values
        self.psi_features_to_drop = []
        
        self.transformer = DropHighPSIFeatures(split_col=self.split_col,
                                               cut_off=self.psi_cut_off,
                                               threshold=self.psi_threshold,
                                               bins=self.psi_bins,
                                               strategy=self.psi_strategy,
                                               missing_values=self.psi_missing_values)

    def fit(self, X, y=None):
        
        self.transformer.fit(X)
        self.psi_features_to_drop = self.transformer.features_to_drop_
        
        if len(self.psi_features_to_drop) > 0:
            log.info(f"Features not passing psi test to drop: {self.psi_features_to_drop}", msg_type="val_tests")
        
        return self
    
    def transform(self, X):
        X = X.drop(columns = self.psi_features_to_drop, axis=1)
        
        return X
   
class CorrFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, corr_ts=0.8, corr_coef_methods=['pearson', 'spearman'], corr_selection_method="missing_values"):
    
        self.corr_coef_methods = corr_coef_methods
        self.corr_selection_method = corr_selection_method
        self.corr_ts = corr_ts
        self.drop_corr_features = []
    
    def fit(self, X, y):
    
        for corr_coef_method in self.corr_coef_methods:
            scs = SmartCorrelatedSelection(threshold=self.corr_ts, method=corr_coef_method, selection_method=self.corr_selection_method)
            scs.fit(X)
            self.drop_corr_features += scs.features_to_drop_
            
        if len(self.drop_corr_features) > 0:
            log.info(f"Corr features to drop: {self.drop_corr_features}", msg_type="preprocessing")
    
        return self
    
class CorrFeaturesTransformerFast(BaseEstimator, TransformerMixin):
    def __init__(self, corr_ts=0.8, corr_coef_methods=['pearson', 'spearman'], corr_selection_method="missing_values"):
    
        self.corr_coef_methods = corr_coef_methods
        self.corr_selection_method = corr_selection_method
        self.corr_ts = corr_ts
        self.drop_corr_features = []
    
    def fit(self, X, y):
    
        for corr_coef_method in self.corr_coef_methods:
            scs = SmartCorrelatedSelectionFast(threshold=self.corr_ts, method=corr_coef_method, selection_method=self.corr_selection_method)
            scs.fit(X)
            self.drop_corr_features += scs.features_to_drop_
            
        if len(self.drop_corr_features) > 0:
            log.info(f"Corr features to drop: {self.drop_corr_features}", msg_type="preprocessing")
    
        return self
 
    def transform(self, X):
        self.drop_corr_features = list(set(self.drop_corr_features))
        X.drop(self.drop_corr_features, axis=1, inplace=True)
 
        return X
 
class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, task_type, target_colname, metric_name, metric_direction, timeout=120, random_state=42,
                 model='lama', strategy='RFA', permutation_n_repeats = 5):
        self.task_type = task_type
        self.target_colname = target_colname
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.timeout = timeout
        self.random_state = random_state
        self.model = model
        self.strategy = strategy
        self.selected_features = []
        self.permutation_n_repeats = permutation_n_repeats
 
        assert self.metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"

    def calc_metric(self, train_with_prediction, test_with_prediction):
        if self.metric_name == 'mae':
            metric_test = mean_absolute_error(y_true=test_with_prediction[self.target_colname], y_pred=test_with_prediction[f'{self.target_colname}_prediction_{self.model}'])
            metric_train = mean_absolute_error(y_true=train_with_prediction[self.target_colname], y_pred=train_with_prediction[f'{self.target_colname}_prediction_{self.model}'])
        if self.metric_name == 'regression_roc_auc_score':
            metric_test = np.round(regression_roc_auc_score(y_true=test_with_prediction[self.target_colname].values, y_pred=test_with_prediction[f'{self.target_colname}_prediction_{self.model}'].values), 4)
            metric_train = np.round(regression_roc_auc_score(y_true=train_with_prediction[self.target_colname].values, y_pred=train_with_prediction[f'{self.target_colname}_prediction_{self.model}'].values), 4)
       
        return metric_train, metric_test
 
    def train_lama_model(self, train, test=None):
 
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
        if test is not None:
            prediction_test = automl.predict(test).data[:, 0]
            train[f'{self.target_colname}_prediction_{self.model}'] = prediction_train
            test[f'{self.target_colname}_prediction_{self.model}'] = prediction_test
        
        return automl
   
    def recursive_feature_addition(self, train, test):
       
        log.info('Выбрана стратегия отбора признаков RFA', msg_type="feature_selection")
        if self.model == 'lama':
            trained_model = self.train_lama_model(train, test)
            fi_first = trained_model.get_feature_scores()
        metric_train, metric_test = self.calc_metric(train, test)
        if f'{self.target_colname}_prediction_{self.model}' in train.columns:
            train.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
            test.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
        cnt_iters_from_last_best_metric_up = 0
        best_features_all = list(fi_first.sort_values('Importance', ascending=False).iloc[:]['Feature'])
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
        upd_cnt = 0
 
        log.info(f'Метрика на всех фичах: {metric_test}', msg_type="feature_selection")
        for i in range(1, len(best_features_all)):
            iter_lst.append(i)
            feature = best_features_all[i]
            train_iter = train[best_features_iter + [feature] + [self.target_colname]]
            test_iter = test[best_features_iter + [feature] + [self.target_colname]]
            if self.model == 'lama':
                trained_model = self.train_lama_model(train_iter, test_iter)
                lama_fi = trained_model.get_feature_scores()
            metric_train, metric_test = self.calc_metric(train_iter, test_iter)
            if f'{self.target_colname}_prediction_{self.model}' in train_iter.columns:
                train_iter.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
                test_iter.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
            test_metric_lst[i] = metric_test
            train_metric_lst[i] = metric_train
            metric_diff_train[i] = train_metric_lst[i] - train_metric_lst[i-1]
            metric_diff_test[i] = test_metric_lst[i] - test_metric_lst[i-1]
 
            if self.metric_direction == 'maximize':
                if metric_test > best_test_metric:
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    best_features_iter.append(feature)
                    upd_cnt += 1
                    log.info(f'Обновлено значение лучшей метрики: {best_test_metric}', msg_type="feature_selection")
                else:
                    drop_features_iter.append(feature)
                    cnt_iters_from_last_best_metric_up += 1
            elif self.metric_direction == 'minimize':
                if metric_test < best_test_metric:
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    best_features_iter.append(feature)
                    upd_cnt += 1
                    log.info(f'Обновлено значение лучшей метрики: {best_test_metric}', msg_type="feature_selection")
                else:
                    cnt_iters_from_last_best_metric_up += 1
                    drop_features_iter.append(feature)
            else:
                assert self.metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"
            best_features_dict[i] = best_features_iter
            drop_features_dict[i] = drop_features_iter
        if upd_cnt == 0:
            best_features_iter = best_features_dict[0]
            best_test_metric = test_metric_lst[0]
        log.info(f'Отобрано {len(best_features_iter)} признаков: {best_features_iter}', msg_type="feature_selection")
        log.info(f'Лучшая метрика на тесте: {best_test_metric}', msg_type="feature_selection")
        feature_selection_dict = {'iter_lst':iter_lst, 'train_metric_lst':train_metric_lst,
                                'test_metric_lst':test_metric_lst, 'features':best_features_dict,
                                'metric_diff_train':metric_diff_train, 'metric_diff_test':metric_diff_test}
       
        return best_features_iter
 
    def recursive_feature_elimination(self, train, test):
       
        if self.model == 'lama':
            trained_model = self.train_lama_model(train, test)
            fi_first = trained_model.get_feature_scores()
        metric_train, metric_test = self.calc_metric(train, test)
        if f'{self.target_colname}_prediction_{self.model}' in train.columns:
            train.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
            test.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
        cnt_iters_from_last_best_metric_up = 0
        best_features_all = list(fi_first.sort_values('Importance', ascending=True).iloc[:]['Feature'])
        best_features_iter = best_features_all.copy()
        drop_features_iter = []
        iter_lst = [0]
        best_features_dict = {0:best_features_iter}
        drop_features_dict = {}
        test_metric_lst = {0:metric_test}
        train_metric_lst = {0:metric_train}
        best_test_metric = 0
        metric_diff_train = {0:0}
        metric_diff_test = {0:0}
        log.info(f'Метрика на всех фичах: {metric_test}', msg_type="feature_selection")
        i = 1
        upd_cnt = 0
        for feature in best_features_all:
            iter_lst.append(i)
            best_features_iter.remove(feature)
            train_iter = train[best_features_all + [self.target_colname]]
            test_iter = test[best_features_all + [self.target_colname]]
            if self.model == 'lama':
                trained_model = self.train_lama_model(train_iter, test_iter)
                lama_fi = trained_model.get_feature_scores()
            metric_train, metric_test = self.calc_metric(train_iter, test_iter)
            if f'{self.target_colname}_prediction_{self.model}' in train.columns:
                train_iter.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
                test_iter.drop(f'{self.target_colname}_prediction_{self.model}', axis=1, inplace=True)
            test_metric_lst[i] = metric_test
            train_metric_lst[i] = metric_train
            metric_diff_train[i] = train_metric_lst[i] - train_metric_lst[i-1]
            metric_diff_test[i] = test_metric_lst[i] - test_metric_lst[i-1]
 
            if self.metric_direction == 'maximize':
                if metric_test > best_test_metric:
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    drop_features_iter.append(feature)
                    upd_cnt += 1
                    log.info(f'Обновлено значение лучшей метрики: {best_test_metric}', msg_type="feature_selection")
                else:
                    best_features_iter.append(feature)
                    cnt_iters_from_last_best_metric_up += 1
            elif self.metric_direction == 'minimize':
                if metric_test < best_test_metric:
                    best_test_metric = metric_test
                    cnt_iters_from_last_best_metric_up = 0
                    drop_features_iter.append(feature)
                    upd_cnt += 1
                    log.info(f'Обновлено значение лучшей метрики: {best_test_metric}', msg_type="feature_selection")
                else:
                    cnt_iters_from_last_best_metric_up += 1
                    best_features_iter.append(feature)
            else:
                assert self.metric_direction in ('maximize', 'minimize'), "Incorrect metric direction.Choose 'maximize' or 'minimize' direction"
            best_features_dict[i] = best_features_iter
            drop_features_dict[i] = drop_features_iter
            i += 1
        if upd_cnt == 0:
            best_features_iter = best_features_dict[0]
            best_test_metric = test_metric_lst[0]
        log.info(f'Отобрано {len(best_features_iter)} признаков: {best_features_iter}', msg_type="feature_selection")
        log.info(f'Лучшая метрика на тесте: {best_test_metric}', msg_type="feature_selection")
        feature_selection_dict = {'iter_lst':iter_lst, 'train_metric_lst':train_metric_lst,
                                'test_metric_lst':test_metric_lst, 'features':best_features_dict,
                                'metric_diff_train':metric_diff_train, 'metric_diff_test':metric_diff_test}
       
        return best_features_iter

    def _weights_scorer(self, trained_model, X, y):
        preds = trained_model.predict(X).data[:, 0]
        if self.metric_name == 'mae':
            return mean_absolute_error(y_true=y, y_pred=preds)
        if self.metric_name == 'regression_roc_auc_score':
            return np.round(regression_roc_auc_score(y_true=y.values, y_pred=preds), 4)

    def _calculate_permutation_scores(self,
        estimator,
        X,
        y,
        col_idx,
        random_state,
        n_repeats,
        max_samples,
    ):
        """Calculate score when `col_idx` is permuted."""
        random_state = check_random_state(random_state)
 
        if max_samples < X.shape[0]:
            row_indices = _generate_indices(
                random_state=random_state,
                bootstrap=False,
                n_population=X.shape[0],
                n_samples=max_samples,
            )
            X_permuted = _safe_indexing(X, row_indices, axis=0)
            y = _safe_indexing(y, row_indices, axis=0)
        else:
            X_permuted = X.copy()
        scores = []
        shuffling_idx = np.arange(X_permuted.shape[0])
        for _ in range(n_repeats):
            random_state.shuffle(shuffling_idx)
            if hasattr(X_permuted, "iloc"):
                col = X_permuted.iloc[shuffling_idx, col_idx]
                col.index = X_permuted.index
                X_permuted[X_permuted.columns[col_idx]] = col
            else:
                X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
            scores.append(self._weights_scorer(estimator, X_permuted, y))
        if isinstance(scores[0], dict):
            scores = _aggregate_score_dicts(scores)
        else:
            scores = np.array(scores)
        return scores

    def _create_importances_bunch(self, baseline_score, permuted_score):
        importances = baseline_score - permuted_score
        return Bunch(
            importances_mean=np.mean(importances, axis=1),
            importances_std=np.std(importances, axis=1),
            importances=importances,
        )
 
    def permutation_importance(self,
        train,
        test,
        n_repeats=5,
        n_jobs=None,
        random_state=None,
        max_samples=1.0,
    ):
        estimator = self.train_lama_model(train)
        X = test.drop(self.target_colname, axis=1)
        y = test[self.target_colname]
        if not hasattr(X, "iloc"):
            X = check_array(X, force_all_finite="allow-nan", dtype=None)
 
        random_state = check_random_state(random_state)
        random_seed = random_state.randint(np.iinfo(np.int32).max + 1)
        if not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])
        elif max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        baseline_score = self._weights_scorer(estimator, X, y)
        log.info(f'Метрика на всех фичах: {baseline_score}', msg_type="feature_selection")
 
        scores = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_permutation_scores)(
                estimator,
                X,
                y,
                col_idx,
                random_seed,
                n_repeats,
                max_samples,
            )
            for col_idx in range(X.shape[1])
        )
 
        if isinstance(baseline_score, dict):
            return {
                name: self._create_importances_bunch(
                    baseline_score[name],
                    # unpack the permuted scores
                    np.array([scores[col_idx][name] for col_idx in range(X.shape[1])]),
                )
                for name in baseline_score
            }
        else:
            return self._create_importances_bunch(baseline_score, np.array(scores))

    def permutation_feature_selection(self, train, test):
        perm = self.permutation_importance(train, test)
        importance = pd.DataFrame(
            {
                "importance_mean": perm["importances_mean"],
                "importance_std": perm["importances_std"]
            },
            index=train.drop(self.target_colname, axis=1).columns
        )
        importance_threshold = importance[importance["importance_mean"] > 0]["importance_mean"].mean() - importance[importance["importance_std"] > 0]["importance_std"].mean()
        best_features_iter = list(importance[importance["importance_mean"] >= importance_threshold].index)
        if best_features_iter == []:
            importance_threshold = 0
            best_features_iter = list(importance[importance["importance_mean"] > importance_threshold].index)
        estimator = self.train_lama_model(train[best_features_iter + [self.target_colname]])
        best_test_metric = self._weights_scorer(estimator, test[best_features_iter], test[self.target_colname])
        log.info(f'Метрика на всех фичах: {best_test_metric}', msg_type="feature_selection")
 
        return best_features_iter

    def fit(self, train, test):
        
        if self.strategy == 'RFA':
            log.info('Выбрана стратегия отбора признаков RFA', msg_type="feature_selection")
            best_features_iter = self.recursive_feature_addition(train, test)
        if self.strategy == 'RFE':
            log.info('Выбрана стратегия отбора признаков RFE', msg_type="feature_selection")
            best_features_iter = self.recursive_feature_elimination(train, test)
        if self.strategy == 'PFI':
            log.info('Выбрана стратегия отбора признаков PFI', msg_type="feature_selection")
            best_features_iter = self.permutation_feature_selection(train, test)
        self.selected_features = best_features_iter

        return self

    def transform(self, X):

        X = X[self.selected_features + [self.target_colname]]
       
        return X
    
    
class CatboostShapFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_features_to_select=50,
        complexity="Regular",
        steps=10,
        random_state=42,
        n_jobs=1):
        """Perform feature selection by recurcive catboost shap.

        Args:
            n_features_to_select (int, optional). Defaults to 50.
            complexity (str, optional): One of ["Approximate", "Regular", "Exact"]. Defaults to "Regular".
            steps (int, optional). Defaults to 10.
            random_state (int, optional). Defaults to 42.
            n_jobs (int, optional). Defaults to 1.
        """
        
        self.n_features_to_select = n_features_to_select
        self.complexity = EShapCalcType[complexity]
        self.steps = steps
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X, y, categorical_features=[]):
        X_train, X_val, y_train, y_val = train_test_split(X.copy(), y.copy(), stratify=y, test_size=0.3, random_state=self.random_state)
        
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)
        
        model = CatBoostClassifier(random_state=self.random_state, verbose=0, early_stopping_rounds=200, iterations=2500,
                                   thread_count=self.n_jobs, allow_writing_files=False)
        
        summary = model.select_features(train_pool, eval_set=val_pool,
                                features_for_select=X_train.columns.tolist(),
                                num_features_to_select=self.n_features_to_select,
                                verbose=False,
                                train_final_model=False,
                                #logging_level="Silent",
                                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                                shap_calc_type=self.complexity, steps=self.steps)
        
        self.selected_features = summary["selected_features_names"]
        
        log.info(f'Selected features: {self.selected_features}', msg_type="feature_selection")
        
        return self
    
    def transform(self, X, y=None, **kwargs):
        
        X = X.copy().loc[:, self.selected_features]
        return X
        