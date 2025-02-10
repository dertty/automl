from typing import List, Optional, Union
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import(
    OneHotEncoder,
    TargetEncoder, 
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer
)
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropHighPSIFeatures
from .selectors import NanFeatureSelector, QConstantFeatureSelector, ObjectColumnsSelector
from .transformers import AdversarialTestTransformer, CorrFeaturesTransformer, DropHighPSITransformer, CorrFeaturesTransformerFast, WinsorizerFast
from ..loggers import get_logger, catchstdout


log = get_logger(__name__)


class PreprocessingPipeline(Pipeline):
    def __init__(
        self, 
        pipe_steps: Union[List[str], str]  = 'all',       
        nan_share_ts: float = 0.2, 
        most_frequent_value_ratio_ts: float = 0.95, 
        impute_num_strategy: str = 'median',
        impute_cat_strategy: str = 'most_frequent',
        outlier_capping_method: str = 'gaussian', 
        outlier_cap_tail: str = 'both',
        corr_ts: float = 0.8, 
        corr_coef_methods: List[str] = ['pearson', 'spearman'],
        corr_selection_method: str = "missing_values", 
        oe_min_freq: float = 0.1,
        cat_encoder_types: List[str] = ['oe', 'ohe', 'mte'],
        num_encoder_types: List[str] = ['ss'],
        num_encoder: str = "ss",
        random_state: int = 42,
        verbose: bool = True,
        ):
        """
        Initialize the PreprocessingPipeline.

        Args:
            pipe_steps (List[str]): 
                List of pipeline steps to include. 
                Defaults to 'all' which equivalent to ['nan_cols_dropper', 'nan_imputer', 'qconst_dropper', 'corr_cols_dropper', 'outlier_capper', 'feature_encoder].     
            nan_share_ts (float): Threshold for the share of NaN values in a column. Defaults to 0.2.
            qconst_feature_val_share_ts (float): Threshold for quasi-constant feature value share. Defaults to 0.95.
            impute_num_strategy (str): Strategy for imputing numerical values. Defaults to 'median'.
            impute_cat_strategy (str): Strategy for imputing categorical values. Defaults to 'most_frequent'.
            outlier_capping_method (str): Method for capping outliers. Defaults to 'gaussian'.
            outlier_cap_tail (str): Tail to cap outliers. Defaults to 'both'.
            corr_ts (float): Correlation threshold. Defaults to 0.8.
            corr_coef_methods (List[str]): List of correlation coefficient methods. Defaults to ['pearson', 'spearman'].
            corr_selection_method (str): Method for selecting correlated features. Defaults to "missing_values".
            oe_min_freq (float): Minimum frequency for ordinal encoding. Defaults to 0.1.
            obj_encoders (List[str]): List of object encoders. Defaults to ['oe', 'ohe', 'mte'].
            num_encoder (str): Numerical encoder. One of ["ss", "quant", "min_max"]. Defaults to "ss".
            random_state (int): Random state for reproducibility. Defaults to 42.
            verbose (bool): Verbosity of the pipeline. Defaults to True.
        """
        super().__init__(steps=[], verbose=verbose)

        self.pipe_steps = pipe_steps
        self.nan_share_ts = nan_share_ts
        self.most_frequent_value_ratio_ts = most_frequent_value_ratio_ts
        self.impute_num_strategy = impute_num_strategy
        self.impute_cat_strategy = impute_cat_strategy
        self.outlier_capping_method = outlier_capping_method
        self.outlier_cap_tail = outlier_cap_tail
        self.corr_ts = corr_ts
        self.corr_coef_methods = corr_coef_methods
        self.corr_selection_method = corr_selection_method
        self.oe_min_freq = oe_min_freq
        self.cat_encoder_types = cat_encoder_types
        self.num_encoder_types = num_encoder_types
        self.num_encoder = num_encoder
        self.random_state = random_state
        
        self._initialize_pipeline_steps()
    
    def _initialize_pipeline_steps(self):
        pipe_steps_dict = OrderedDict([
            ("nan_cols_dropper", ("nan_cols_dropper", self.nan_col_selector(nan_share_ts=self.nan_share_ts))),
            ("nan_imputer", ("nan_imputer", self.nan_imputer(impute_num_strategy=self.impute_num_strategy, impute_cat_strategy=self.impute_cat_strategy))),
            ("qconst_dropper", ("qconst_dropper", self.qconst_col_selector(most_frequent_value_ratio_ts=self.most_frequent_value_ratio_ts))),
            ("corr_cols_dropper", ("corr_cols_dropper", CorrFeaturesTransformerFast(
                corr_ts=self.corr_ts, 
                corr_coef_methods=self.corr_coef_methods,
                corr_selection_method=self.corr_selection_method,
                ))),
            ("outlier_capper", ("outlier_capper", self.outlier_capper(outlier_capping_method=self.outlier_capping_method, outlier_cap_tail=self.outlier_cap_tail))),
            ("feature_encoder", ("feature_encoder", self.object_encoder(
                cat_encoder_types=self.cat_encoder_types, 
                num_encoder_types=self.num_encoder_types, 
                oe_min_freq=self.oe_min_freq, 
                random_state=self.random_state))),
        ])
        
        if isinstance(self.pipe_steps, str) and self.pipe_steps == 'all':
            self.steps = [pipe_steps_dict[pipe_step] for pipe_step in pipe_steps_dict.keys()]
        elif isinstance(self.pipe_steps, iter):
            self.steps = []
            for pipe_step in self.pipe_steps:
                assert pipe_step in pipe_steps_dict.keys(), f"Unknown pipe step: {pipe_step}"
                self.steps.append(pipe_steps_dict[pipe_step])
    
    @catchstdout(log)
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)
    
    @catchstdout(log)
    def fit_transform(self, *args, **kwargs):
        return super().fit_transform(*args, **kwargs)
    
    @catchstdout(log)
    def transform(self, *args, **kwargs):
        return super().transform(*args, **kwargs)
    
    @staticmethod
    def nan_col_selector(nan_share_ts: float = 0.0):
        # Трансформер для отбора признаков с долей пропусков менее заданного значения
        nan_col_selector = ColumnTransformer(
            transformers=[('DropNanColumns', 'drop', NanFeatureSelector(nan_share_ts=nan_share_ts))],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
            ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
        return nan_col_selector
    
    @staticmethod
    def nan_imputer(impute_num_strategy: str = 'median', impute_cat_strategy: str = 'most_frequent'):
        # Трансформер для заполнения пропусков
        nan_imputer = ColumnTransformer(
            transformers=[
                ('impute_num', SimpleImputer(strategy=impute_num_strategy), make_column_selector(dtype_include='number')),
                ('impute_cat', SimpleImputer(strategy=impute_cat_strategy), make_column_selector(dtype_exclude='number')),
                ],
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
            ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
        return nan_imputer
    
    @staticmethod
    def qconst_col_selector(most_frequent_value_ratio_ts: float = 0.95):
        # Трансформер для отбора (квази)константных признаков
        qconst_col_selector = ColumnTransformer(
            transformers=[('DropQConstantColumns', 'drop', QConstantFeatureSelector(feature_val_share_ts=most_frequent_value_ratio_ts))],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
            ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
        return qconst_col_selector
    
    @staticmethod
    def outlier_capper(outlier_capping_method: str = 'gaussian', outlier_cap_tail: str = 'both'):
        # Трансформер для ограничения выбросов
        outlier_capper = ColumnTransformer(
            transformers=[
                (
                    'outliers_capping', 
                    WinsorizerFast(capping_method=outlier_capping_method, tail=outlier_cap_tail, missing_values='ignore'), 
                    make_column_selector(dtype_include='number')
                ),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
            ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
        return outlier_capper

    @staticmethod
    def object_encoder(cat_encoder_types: List[str] = ['ohe', 'oe', 'mte'], num_encoder_types: List[str] = ['ss', 'quant', 'min_max'], oe_min_freq: float = 0.1, random_state: int = 0):
        # Трансформер для кодирования категориальных признаков
        cat_encoders_dict = {
            'ohe': (
                'OneHotEncoder', 
                OneHotEncoder(
                    sparse_output=False, 
                    drop='first', 
                    handle_unknown='ignore', 
                    dtype=np.int16), 
                ObjectColumnsSelector(mode='ohe')),
            'oe': (
                'OrdinalEncoder', 
                OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    encoded_missing_value=-1, 
                    unknown_value=-1, 
                    min_frequency=oe_min_freq, 
                    dtype=np.int16), 
                ObjectColumnsSelector(mode='oe')),
            'mte': (
                'MeanTargetEncoder', 
                TargetEncoder(target_type='auto'), 
                ObjectColumnsSelector(mode='mte'))
            }
        cat_transformers = [cat_encoders_dict[encoder] for encoder in cat_encoder_types]
        
        num_encoders_dict = {
            'ss': ('StandardScaler', StandardScaler(), make_column_selector(dtype_include="number")),
            'quant':('QuantileTransformer', QuantileTransformer(random_state=random_state), make_column_selector(dtype_include="number")),
            'min_max': ('MinMaxScaler', MinMaxScaler(clip=True), make_column_selector(dtype_include="number")),
            }
        num_transformers = [num_encoders_dict[encoder] for encoder in num_encoder_types]
        
        feature_encoders = ColumnTransformer(
            transformers=cat_transformers + num_transformers,
            remainder='passthrough',
            verbose_feature_names_out=True
            ).set_output(transform='pandas')
        
        return feature_encoders
    
class ValTestsPipeline(Pipeline):

    def __init__(
        self, 
        pipe_steps: List[str] = ['all'], 
        split_col: str = 'is_test_for_val', 
        psi_cut_off: float = 0.5, 
        psi_threshold: float = 0.2,
        psi_bins: int = 15, 
        psi_strategy: str = 'equal_width', 
        adversarial_auc_trshld: float = 0.7, 
        verbose: bool = True,
        random_state: int = 42,
        ):
        super().__init__(steps=[], verbose=verbose)
       
        self.pipe_steps = pipe_steps
        self.random_state = random_state
        self.split_col = split_col
        self.psi_cut_off = psi_cut_off
        self.psi_threshold = psi_threshold
        self.psi_bins = psi_bins
        self.psi_strategy = psi_strategy
        self.adversarial_auc_trshld = adversarial_auc_trshld
 
        pipe_steps_dict = {
            "PSI_test":("PSI_test", DropHighPSITransformer(
                split_col=self.split_col, 
                psi_cut_off=self.psi_cut_off, 
                psi_threshold=self.psi_threshold, 
                psi_bins=self.psi_bins, 
                psi_strategy=self.psi_strategy, 
                psi_missing_values='ignore'
                )),
            "Adversarial_test":("Adversarial_test", AdversarialTestTransformer(
                split_col=self.split_col, 
                auc_trshld = self.adversarial_auc_trshld,
                random_state=self.random_state, 
                )),
        }
        if self.pipe_steps == ['all']:
            log.info('Успешно заданы шаги pipeline', msg_type="val_tests")
        else:
            if len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) == 0:
                log.info('Успешно заданы шаги pipeline', msg_type="val_tests")
            else:
                log.info(f'Необходимо переопределить шаги pipeline, удалите шаги {set(self.pipe_steps) - set(pipe_steps_dict.keys())}', msg_type="val_tests")
                assert len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) > 0, 'Incorrect pipe steps'
        pipe_steps_lst_of_tuples = [pipe_steps_dict[pipe_step] for pipe_step in (self.pipe_steps \
            if self.pipe_steps[0] != 'all' else pipe_steps_dict.keys())]
   
        self.steps = pipe_steps_lst_of_tuples
        
    @catchstdout(log)
    def fit(self, X_train, X_test, **kwargs):
        # explicitly add a split column to data
        X_train = X_train.copy()
        X_train[self.split_col] = 0
        
        X_test = X_test.copy()
        X_test[self.split_col] = 1
        
        # construct single data frame
        X = pd.concat([X_train, X_test], ignore_index=True)
        return super().fit(X, **kwargs)
    
    @catchstdout(log)
    def fit_transform(self, X_train, X_test, **kwargs):
        # explicitly add a split column to data
        X_train = X_train.copy()
        X_train[self.split_col] = 0
        
        X_test = X_test.copy()
        X_test[self.split_col] = 1
        
        # construct single data frame
        X = pd.concat([X_train, X_test], ignore_index=True)
        
        X_transformed = super().fit_transform(X, **kwargs)
        
        # return only the train part of the data 
        # and drop self.split_col
        return X_transformed.loc[X_transformed[self.split_col] == 0].drop(columns=self.split_col).reset_index(drop=True)
    
    @catchstdout(log)
    def transform(self, X_test, **kwargs):
        X = X_test.copy()
        X[self.split_col] = 1
        
        X_transformed = super().transform(X, **kwargs)
        
        # drop self.split_col from data
        return X_transformed.drop(columns=self.split_col)