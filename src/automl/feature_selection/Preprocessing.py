import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropHighPSIFeatures
from catboost import CatBoostClassifier, Pool

def split_data(df, index_columns, target_column, time_column=None, test_size=0.3, random_state = 42):
    """
    Разделение выборок для обучения и тестирования.

    Args:
        df (pd.DataFrame): Датафрейм для разделения.
        index_columns (lst): Индекс - переменные
        target_column (str): Целевая переменная.
        time_column (str): Переменная с датой.
        test_size (float): Доля тестовой выборки из общего датафрейма.
        random_state (int): random_state.
    Returns:
        train (pd.DataFrame): Данные для обучения.
        test (pd.DataFrame): Данные для тестирования.
    """
    if time_column:
        index_columns.append(time_column)
        df_oot, df_oos = train_test_split(df, test_size = 0.5, stratify=df[time_column], random_state=random_state)
        df_oot['one_row_score'] = 1
        df_oot['cum_share_data'] = (df_oot.sort_values(time_column)[['one_row_score']].cumsum()/df_oot[['one_row_score']].sum())
        df_oot.drop('one_row_score', axis=1, inplace=True)
        train_oot = df_oot[df_oot['cum_share_data'] < 1 - test_size]
        train_oot.drop('cum_share_data', axis=1, inplace=True)
        test_oot = df_oot[df_oot['cum_share_data'] >= 1 - test_size]
        test_oot.drop('cum_share_data', axis=1, inplace=True)
        test_oot['is_test'] = 2
        train_oos, test_oos = train_test_split(df_oos, test_size = test_size, stratify=df_oos[[time_column]], random_state=random_state)
        test_oos['is_test'] = 3
        train = pd.concat([train_oot, train_oos])
        train['is_test'] = 0
        test = pd.concat([test_oot, test_oos])
        print(f'train_share: {train.shape[0]/df.shape[0]}')
        print(f'test_share: {test.shape[0]/df.shape[0]}')
        print(f'test_oos_share: {test[test["is_test"]==3].shape[0]/df.shape[0]}')
        print(f'test_oot_share: {test[test["is_test"]==2].shape[0]/df.shape[0]}')
        print("Среднее значение таргета")
        print(f'train target mean: {train[target_column].mean()}')
        print(f'test_oos_share target mean:   {test[test["is_test"]==3][target_column].mean()}')
        print(f'test_oot_share target mean:  {test[test["is_test"]==2][target_column].mean()}')
    else:
        train, test = train_test_split(df, test_size = test_size, random_state=random_state)
        test['is_test'] = 3
        train['is_test'] = 0
        print("Среднее значение таргета")
        print(f'train target mean: {train[target_column].mean()}')
        print(f'test target mean: {test[target_column].mean()}')
    index_columns.append('is_test')
    train.set_index(index_columns, inplace=True)
    test.set_index(index_columns, inplace=True)

    return train, test

class NanFeatureSelector():
    '''
    Класс для отбора признаков с долей пропусков больше заданного значения.

    Attributes:
        nan_share_ts (float): Пороговое значение доли пропущенных значений.
    '''
    def __init__(self, nan_share_ts=0.2):
        self.nan_share_ts = nan_share_ts
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        X_copy= df.copy()
        nan_share = X_copy.isna().sum() / X_copy.shape[0]
        nan_features = nan_share[nan_share >= self.nan_share_ts].index.tolist()
        
        return nan_features

class QConstantFeatureSelector():
    '''
    Класс для отбора константных и квазиконстантных признаков.

    Attributes:
        feature_val_share_ts (float): Пороговое значение максимальной доли значения среди прочих значений признака.
    '''
    def __init__(self, feature_val_share_ts=0.98):
        self.feature_val_share_ts = feature_val_share_ts

    def find_share_of_value(self, arr, col_name):
        arr_len = len(arr)
        arr_unique = arr.unique()
        if len(arr_unique) / arr_len > (1-self.feature_val_share_ts):
            return None
        arr_value_counts = np.array([arr.value_counts()[x] for x in arr_unique])
        arr_share = arr_value_counts / arr_len
        max_arr_share = np.max(arr_share)
        if max_arr_share >= self.feature_val_share_ts:
            return col_name
        else:
            return None
        
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        X_copy= df.copy()
        qconst_cols = [self.find_share_of_value(X_copy[x], x) for x in X_copy.columns]
        qconst_cols = [col for col in qconst_cols if col]
        
        return qconst_cols

class ObjectColumnsSelector:
    '''
    Класс для отбора категориальных признаков с выбором стратегии кодирования в числовые признаки.

    Attributes:
        ohe_limiter (int): Максимальное число уникальных категорий для выбора стратегии OneHotEncoding.
        mode (str): Стратегия кодирования признаков.
    '''
    def __init__(self, ohe_limiter=5, mode='ohe'):
        self.ohe_limiter = ohe_limiter
        self.mode = mode
        
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        
        df_obj = df.select_dtypes(include='object')
        counter_ = df_obj.nunique()

        if self.mode == 'ohe':
            final_cols = counter_[counter_ <= self.ohe_limiter]
        elif self.mode == 'mte':
            final_cols = counter_[counter_ > self.ohe_limiter]
        else:
            return []
            
        return final_cols.index.tolist()

class PearsonCorrFeatureSelector():
    '''
    Класс для выявления зависимых признаков c помощью коэффициента корреляции Пирсона. \n
    Возвращает вектор названий, в который включен один из двух коррелирующих признаков. \n
    Коэффициент корреляции Пирсона (Pearson correlation coefficient) — это параметрический тест, который строится на основе расчета ковариации двух переменных, 
    разделенного на произведение среднеквадратического отклонения каждой из них.
    Значения, приближающиеся к 1 указывают на сильную положительную линейную корреляцию. Близкие к -1 на сильную отрицательную линейную корреляцию. 
    Околонулевые значения предполагают отсутствие линейной корреляции. \n
    У коэффициента Пирсона есть ряд ограничений, в частности, он выявляет только линейную взаимосвязь количественных переменных. 

    Attributes:
        corr_ts (float): Пороговое значение коэффициента корреляции двух переменных.
    '''
    def __init__(self, corr_ts=0.8):
        self.corr_ts = corr_ts
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        X_copy= df.select_dtypes(include='number').copy()
        corr_matrix = df.corr(method = 'pearson').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_cols = [column for column in upper.columns if any(upper[column] > self.corr_ts)]
        
        return corr_cols
    
class SpearmanCorrFeatureSelector():
    '''
    Класс для выявления зависимых признаков c помощью коэффициента корреляции Спирмена. \n
    Возвращает вектор названий, в который включен один из двух коррелирующих признаков. \n
    Коэффициент ранговой корреляции Спирмена (Spearman’s Rank Correlation Coefficient) хорошо измеряет монотонную
    зависимость двух переменных.
    Это непараметрический тест, который не предполагает каких-либо допущений о распределении генеральной совокупности.

    Attributes:
        corr_ts (float): Пороговое значение коэффициента корреляции двух переменных.
    '''
    def __init__(self, corr_ts=0.8):
        self.corr_ts = corr_ts
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        X_copy= df.select_dtypes(include='number').copy()
        corr_matrix = df.corr(method = 'spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_cols = [column for column in upper.columns if any(upper[column] > self.corr_ts)]
        
        return corr_cols


class ValTestSelector():
    '''
    Класс для отбора признаков, не прошедших тесты\n
    - PSI\n
        PSI test состоит в расчете метрики PSI и анализе диаграмм распределения для поиска бакетов, в которых распределения факторов между выборками значительно изменяются.
    - Adversarial test\n
        Adversarial test проверяет наличие значительных различий в трейне и тесте(нерепрезентативность выборок). 
        Трейн и тест размечаются бинарно. Разметка используется в качестве таргета для построения модели классификации.
        Последовательно удаляются признаки, которые хорошо разделяют трейн и тест (те, которые модель считает наиболее значимыми).
        Удаление признаков происходит до момента достижения моделью качества auc_trshld на метрике Roc Auc.
  
    '''
    def __init__(self, X_train, X_test, psi_ts=0.2):
        self.X_train = X_train
        self.X_test = X_test
        self.psi_ts = psi_ts
        pass

    def cb_feature_importance(self, model):
        '''Feature importance моделей (на текущий момент реализовано для Catboost) с возможностью визуализации результата
            Args:
            model: обученна€ модель
            Returns:
            fi_df: датафрейм с feature importance
        '''
        #Create arrays from feature importance and feature names
        feature_importance = model.feature_importances_
        feature_names = model.feature_names_
        
        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names, 'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)
        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
        
        return fi_df
    def adversarial_test(self, df_train, df_test, random_state=42, auc_trshld = 0.65):
        '''Adversarial test.
            Args:
            df_train: обучающая выборка
            df_test: тестовая выборка
            random_state: random_state
            auc_trshld: граница дл€ метрики
            Returns:
            ignore_features: признаки, не прошедшие adversarial тест
        '''
        df_train['dataset_label'] = 0
        df_test['dataset_label'] = 1
        target = 'dataset_label'

        def create_adversarial_data(df_train, df_test):
            N_val = df_test.shape[0]
            df_master = pd.concat([df_train, df_test], axis=0)
            adversarial_val = df_master.sample(N_val, replace=False)
            adversarial_train = df_master[~df_master.index.isin(adversarial_val.index)]
            return adversarial_train, adversarial_val

        adversarial_train, adversarial_test = create_adversarial_data(df_train, df_test)

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
                'random_seed': random_state,
                'ignored_features': [], 
                'depth': 4
            }
        model = CatBoostClassifier(**params)
        model.fit(train_data, eval_set=holdout_data)
        model_auc = model.best_score_['validation']['AUC']
        cb_feature_importance_df = self.cb_feature_importance(model) 
        top_fi = cb_feature_importance_df.iloc[0]['feature_importance']
        top_fi_name = cb_feature_importance_df.iloc[0]['feature_names']
        while (model_auc > auc_trshld) & (top_fi != 0):
            ignore_features.append(top_fi_name)
            params.update({'ignored_features': ignore_features})
            model = CatBoostClassifier(**params)
            model.fit(train_data, eval_set=holdout_data)
            model_auc = model.best_score_['validation']['AUC']
            cb_feature_importance_df = self.cb_feature_importance(model) 
            top_fi = cb_feature_importance_df.iloc[0]['feature_importance']
            top_fi_name = cb_feature_importance_df.iloc[0]['feature_names']
        
        return ignore_features
    
    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to pandas dataframes"
            )
        X_train_copy= self.X_train.copy()
        X_test_copy= self.X_test.copy()
        X_train_copy['is_test_psi'] = 0
        X_test_copy['is_test_psi'] = 1
        #exclude_by_psi_test = [col for col in X_train_copy.columns if self.calculate_psi(X_train_copy[col], X_test_copy[col]) >= self.psi_ts]
        psi_train_val_vs_test = DropHighPSIFeatures(split_col='is_test_psi', cut_off=0.5, threshold=0.2, bins=15, strategy='equal_width')
        psi_train_val_vs_test.fit(pd.concat([X_train_copy, X_test_copy]))
        exclude_by_psi_test = psi_train_val_vs_test.features_to_drop_
        X_train_copy.drop('is_test_psi', axis=1, inplace=True)
        X_test_copy.drop('is_test_psi', axis=1, inplace=True)
        exclude_by_adversarial_test =  self.adversarial_test(X_train_copy, X_test_copy)
        exclude_cols_by_all_tests = list(set(exclude_by_adversarial_test + exclude_by_psi_test))
        return exclude_cols_by_all_tests

def create_preprocess_pipe(nan_share_ts=0.2, qconst_feature_val_share_ts=0.95, impute_num_strategy='median', 
                           impute_cat_strategy='most_frequent', 
                           outlier_capping_method='gaussian', outlier_cap_tail='both',
                           corr_ts = 0.8):

    # Трансформер для отбора признаков с долей пропусков менее заданного значения 
    nan_col_selector = ColumnTransformer( 
        transformers=[          
            ('DropNanColumns', 'drop', NanFeatureSelector(nan_share_ts=nan_share_ts))        
        ],
        remainder='passthrough',
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

    # Трансформер для отбора (квази)константных признаков
    qconst_col_selector = ColumnTransformer( 
        transformers=[          
            ('DropQConstantColumns', 'drop', QConstantFeatureSelector(feature_val_share_ts=qconst_feature_val_share_ts))        
        ],
        remainder='passthrough',
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

    # Трансформер для заполнения пропусков 
    nan_imputer = ColumnTransformer(
        transformers=[
            ('impute_num', SimpleImputer(strategy=impute_num_strategy), make_column_selector(dtype_include="number")),
            ('impute_cat', SimpleImputer(strategy=impute_cat_strategy), make_column_selector(dtype_exclude="number"))
        ],
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
    # Трансформер для ограничения выбросов
    outlier_capper = ColumnTransformer(
        transformers=[
            ('outliers_capping', Winsorizer(capping_method=outlier_capping_method, tail=outlier_cap_tail), make_column_selector(dtype_include="number")),
        ],
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

    # Трансформер для кодирования категориальных признаков 
    object_encoder = ColumnTransformer(
        transformers=[
            ('OneHotEncoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ObjectColumnsSelector(mode='ohe')),             
            ('MeanTargetEncoder', TargetEncoder(target_type="auto"), ObjectColumnsSelector(mode='mte'))        
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')
    # Трансформер для коррелирующих признаков
    corr_col_selector = ColumnTransformer( 
        transformers=[          
            ('DropPirsonCorrColumns', 'drop', PearsonCorrFeatureSelector(corr_ts=corr_ts)),
            ('DropSpearmanCorrColumns', 'drop', SpearmanCorrFeatureSelector(corr_ts=corr_ts))     
        ],
        remainder='passthrough',
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

    preprocessing_pipe = Pipeline(
    [
        ("nan_cols_dropper", nan_col_selector),
        ("nan_imputer", nan_imputer),
        ("object_encoder", object_encoder),
        ("qconst_dropper", qconst_col_selector),
        ("outlier_capper", outlier_capper),
        ("corr_cols_dropper", corr_col_selector),
    ], verbose=True
    )
    return preprocessing_pipe

def create_test_pipe(X_train_prep, X_test_prep):
    # Трансформер для исключения признаков, не прошедших тесты
    tests_col_selector = ColumnTransformer( 
        transformers=[          
            ('ExcludeByAdversarialAndPsiTests', 'drop', ValTestSelector(X_train_prep, X_test_prep))        
        ],
        remainder='passthrough',
        verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
    ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
    val_test_pipe = Pipeline(
    [
        ("AdversarialAndPsiTests", tests_col_selector),
    ], verbose=True
    )

    return val_test_pipe

def preprocess_data(df_path, index_cols, target_col, time_col=None, test_size=0.15, random_state=42, preprocessing_pipe_path='outputs/preprocessing_pipe.joblib',val_test_pipe_path='outputs/test_pipe.joblib',
                    nan_share_ts=0.2, qconst_feature_val_share_ts=0.95, impute_num_strategy='median', 
                    impute_cat_strategy='most_frequent', 
                    outlier_capping_method='gaussian', outlier_cap_tail='both',
                    corr_ts = 0.8):
    """
    Docstring в разработке
    """
    
    # Считываем данные и удаляем дубли
    if df_path.split('.')[-1] == 'csv':
        df = pd.read_csv(df_path).drop_duplicates()
    elif df_path.split('.')[-1] == 'parquet':
        df = pd.read_parquet(df_path).drop_duplicates()
    else:
        assert df_path.split('.')[-1] in ('csv', 'parquet'), 'Incorrect data format. Export your DataFrame to CSV or Parquet file'
    # Разделяем выборки для обучения и тестирования
    train, test = split_data(df, index_cols, target_col, time_col, test_size, random_state)
    X_train, y_train = train.drop(target_col, axis=1), train[target_col]
    X_test, y_test = test.drop(target_col, axis=1), test[target_col]
    preprocessing_pipe = create_preprocess_pipe(nan_share_ts, qconst_feature_val_share_ts, impute_num_strategy, 
                                                impute_cat_strategy, 
                                                outlier_capping_method, outlier_cap_tail,
                                                corr_ts)
    # Делаем fit пайплайна обработки на тренировочной выборке
    preprocessing_pipe.fit(X_train, y_train)
    joblib.dump(preprocessing_pipe, preprocessing_pipe_path)
    # Делаем transform признаков для обучающей и тестовой выборок
    X_train_prep = preprocessing_pipe.transform(X_train)
    X_test_prep = preprocessing_pipe.transform(X_test)

    print(X_train_prep.shape, X_train_prep.columns)
    print(X_test_prep.shape, X_test_prep.columns)
    val_test_pipe = create_test_pipe(X_train_prep, X_test_prep)
    val_test_pipe.fit(X_train_prep, y_train)
    # joblib.dump(val_test_pipe, val_test_pipe_path)
    X_train_checked= val_test_pipe.transform(X_train_prep)
    X_test_checked = val_test_pipe.transform(X_test_prep)
    train_checked = X_train_checked.join(y_train)
    test_checked = X_test_checked.join(y_test)
    if time_col:
        sort_lst = [time_col] + index_cols
        train_checked.sort_values(by=sort_lst, inplace=True)
        test_checked.sort_values(by=sort_lst, inplace=True)
        return train_checked, test_checked
    else:
        return train_checked, test_checked

#######################################################################################################################################
# fit_pipe = Pipeline(
#     [
#         ("col_imputer", col_imputer()),
#         ("col_transformer_with_selector", col_transformer_with_selector),
#         ("col_transformer", col_transformer_with_selector),
#     ]
# )
# transform_pipe = Pipeline(
#     [
#         ()
#     ]
# )
#######################################################################################################################################
# class CustomTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#     def fit(self, X, y):
#         if not hasattr(X, "iloc"):
#             raise ValueError(
#                 "CustomTransformer can only be applied to pandas dataframes in X argument"
#             )
#         ### Застрахуемся от inplace изменений
#         X_copy= X.copy()
#         y_copy = y.copy()
#         # Обновляем атрибуты self, например,
#         #self.nan_features = nan_share[nan_share > self.nan_limiter].index.tolist()

#         return self

#     def transform(self, X):
#         if not hasattr(X, "iloc"):
#             raise ValueError(
#                 "CustomTransformer can only be applied to pandas dataframes in X argument"
#             )
#         X_copy= X.copy()

#         ### Трансформируем X
#         X_copy.drop(self.nan_features, axis=1, inplace=True)

#         return X_copy
#######################################################################################################################################
