import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropHighPSIFeatures
from automl.feature_selection.selectors import NanFeatureSelector, QConstantFeatureSelector, ObjectColumnsSelector
from automl.feature_selection.transformers import AdversarialTestTransformer, CorrFeaturesTransformer
from automl.loggers import get_logger
log = get_logger(__name__)

class PreprocessingPipeline(Pipeline):
    def __init__(self, pipe_steps = ['all'],
                 nan_share_ts=0.2, qconst_feature_val_share_ts=0.95, impute_num_strategy='median',
                 impute_cat_strategy='most_frequent',
                 outlier_capping_method='gaussian', outlier_cap_tail='both',
                 corr_ts = 0.8, corr_coef_methods=['pearson', 'spearman'],
                 corr_selection_method="missing_values", oe_min_freq=0.1,
                 obj_encoders = ['oe', 'ohe', 'mte'], verbose=True):

        self.pipe_steps = pipe_steps
        self.nan_share_ts = nan_share_ts
        self.qconst_feature_val_share_ts = qconst_feature_val_share_ts
        self.impute_num_strategy = impute_num_strategy
        self.impute_cat_strategy = impute_cat_strategy
        self.outlier_capping_method = outlier_capping_method
        self.outlier_cap_tail = outlier_cap_tail
        self.corr_ts = corr_ts
        self.corr_coef_methods = corr_coef_methods
        self.corr_selection_method = corr_selection_method
        self.oe_min_freq = oe_min_freq
        self.obj_encoders = obj_encoders
        self.verbose=verbose
        self.memory=None

        # Трансформер для отбора признаков с долей пропусков менее заданного значения
        nan_col_selector = ColumnTransformer(
            transformers=[         
                ('DropNanColumns', 'drop', NanFeatureSelector(nan_share_ts=self.nan_share_ts))       
            ],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
        ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

        # Трансформер для отбора (квази)константных признаков
        qconst_col_selector = ColumnTransformer(
            transformers=[         
                ('DropQConstantColumns', 'drop', QConstantFeatureSelector(feature_val_share_ts=self.qconst_feature_val_share_ts))
            ],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
        ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
 
        # Трансформер для заполнения пропусков
        nan_imputer = ColumnTransformer(
            transformers=[
                ('impute_num', SimpleImputer(strategy=self.impute_num_strategy), make_column_selector(dtype_include='number')),
                ('impute_cat', SimpleImputer(strategy=self.impute_cat_strategy), make_column_selector(dtype_exclude='number'))
            ],
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
        ).set_output(transform='pandas')      # Трансформер будет возвращать pandas
        # Трансформер для ограничения выбросов
        outlier_capper = ColumnTransformer(
            transformers=[
                ('outliers_capping', Winsorizer(capping_method=self.outlier_capping_method, tail=self.outlier_cap_tail, missing_values='ignore'), make_column_selector(dtype_include='number')),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False   # Оставляем оригинальные названия колонок
        ).set_output(transform='pandas')      # Трансформер будет возвращать pandas

        # Трансформер для кодирования категориальных признаков
        obj_encoders_dict = {
            'ohe':('OneHotEncoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', dtype=np.int16), ObjectColumnsSelector(mode='ohe')),
            'oe':('OrdinalEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', encoded_missing_value=-1, unknown_value=-1, min_frequency=self.oe_min_freq, dtype=np.int16), ObjectColumnsSelector(mode='oe')),
            'mte':('MeanTargetEncoder', TargetEncoder(target_type='auto'), ObjectColumnsSelector(mode='mte'))
            }
        if self.pipe_steps[0] == 'all' or 'object_encoder' in self.pipe_steps:
            obj_transformers = [obj_encoders_dict[obj_encoder] for obj_encoder in self.obj_encoders]
        object_encoder = ColumnTransformer(
            transformers=obj_transformers,
            remainder='passthrough',
            verbose_feature_names_out=True
        ).set_output(transform='pandas')
        pipe_steps_dict = {
            "nan_cols_dropper":("nan_cols_dropper", nan_col_selector),
            "outlier_capper":("outlier_capper", outlier_capper),
            "nan_imputer":("nan_imputer", nan_imputer),
            "corr_cols_dropper":("corr_cols_dropper", CorrFeaturesTransformer(corr_ts=self.corr_ts, corr_coef_methods=self.corr_coef_methods,
                                                          corr_selection_method=self.corr_selection_method)),
            "object_encoder":("object_encoder", object_encoder),
            "qconst_dropper":("qconst_dropper", qconst_col_selector)
        }
        if self.pipe_steps == ['all']:
            log.info('Успешно заданы шаги pipeline', msg_type="preprocessing")
        else:
            if len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) == 0:
                log.info('Успешно заданы шаги pipeline', msg_type="preprocessing")
            else:
                log.info(f'Необходимо переопределить шаги pipeline, удалите шаги {set(self.pipe_steps) - set(pipe_steps_dict.keys())}', msg_type="preprocessing")
                assert len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) > 0, 'Incorrect pipe steps'
        pipe_steps_lst_of_tuples = [pipe_steps_dict[pipe_step] for pipe_step in (self.pipe_steps if self.pipe_steps[0] != 'all' else pipe_steps_dict.keys())]
        self.steps = pipe_steps_lst_of_tuples

class ValTestsPipeline(Pipeline):

    def __init__(self, pipe_steps = ['all'], random_state=42,
                 split_col='is_test_for_val', psi_cut_off=0.5, psi_threshold=0.2,
                 psi_bins=15, psi_strategy='equal_width', adversarial_auc_trshld=0.7, verbose=True):
       
        self.pipe_steps = pipe_steps
        self.random_state = random_state
        self.split_col = split_col
        self.psi_cut_off = psi_cut_off
        self.psi_threshold = psi_threshold
        self.psi_bins = psi_bins
        self.psi_strategy = psi_strategy
        self.adversarial_auc_trshld = adversarial_auc_trshld
        self.verbose=verbose
        self.memory=None
 
        pipe_steps_dict = {
            "PSI_test":("PSI_test", DropHighPSIFeatures(split_col=self.split_col, cut_off=self.psi_cut_off, threshold=self.psi_threshold, bins=self.psi_bins, strategy=self.psi_strategy, missing_values='ignore')),
            "Adversarial_test":("Adversarial_test", AdversarialTestTransformer(split_col=self.split_col, random_state=self.random_state, auc_trshld = self.adversarial_auc_trshld)),
        }
        if self.pipe_steps == ['all']:
            log.info('Успешно заданы шаги pipeline', msg_type="val_tests")
        else:
            if len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) == 0:
                log.info('Успешно заданы шаги pipeline', msg_type="val_tests")
            else:
                log.info(f'Необходимо переопределить шаги pipeline, удалите шаги {set(self.pipe_steps) - set(pipe_steps_dict.keys())}', msg_type="val_tests")
                assert len(set(self.pipe_steps) - set(pipe_steps_dict.keys())) > 0, 'Incorrect pipe steps'
        pipe_steps_lst_of_tuples = [pipe_steps_dict[pipe_step] for pipe_step in (self.pipe_steps if self.pipe_steps[0] != 'all' else pipe_steps_dict.keys())]
   
        self.steps = pipe_steps_lst_of_tuples