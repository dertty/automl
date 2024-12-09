import numpy as np

from .feature_processing import (
    CatboostShapFeatureSelector,
    FeatureSelectionTransformer,
    PreprocessingPipeline,
    ValTestsPipeline,
)
from .loggers import enable_logging_to_file
from .model import AutoModel

FEATURE_SELECTIONS_MAPPING = {
    "CatboostByShap": CatboostShapFeatureSelector,
    "LAMA": FeatureSelectionTransformer,
}


class AutoML:
    """AutoML - automate routine ML tasks in 5 lines of code.

    Parameters
    ----------
    task
        Machine Learning task to solve, by default "classification"
        Should be one of:
            - "classification" (binary and multiclass)
            - "regression"
    use_preprocessing_pipeline, optional
        Whether to use preprocessing pipeline, by default True
    preprocessing_pipeline_kwargs, optional
        Keyword arguments to initialize preprocessing_pipeline, by default {}
        List of possible arguments and their default values:
            - pipe_steps = ['all']
            - nan_share_ts=0.2
            - qconst_feature_val_share_ts=0.95
            - impute_num_strategy='median'
            - impute_cat_strategy='most_frequent'
            - outlier_capping_method='gaussian'
            - outlier_cap_tail='both'
            - corr_ts = 0.8
            - corr_coef_methods=['pearson', 'spearman']
            - corr_selection_method="missing_values"
            - oe_min_freq=0.1
            - obj_encoders = ['oe', 'ohe', 'mte']
            - num_encoder = "ss"
            - verbose=True
    use_val_test_pipeline, optional
        Whether to use val_test_pipeline, by default True
    val_test_pipeline_kwargs, optional
        Keyword arguments to initialize val_test_pipeline, by default {}
        List of possible arguments and their default values:
            - pipe_steps = ['all']
            - split_col='is_test_for_val'
            - psi_cut_off=0.5
            - psi_threshold=0.2
            - psi_bins=15
            - psi_strategy='equal_width'
            - adversarial_auc_trshld=0.7
            - verbose=True
    feature_selector_type, optional
        Whether to use feature_selector and which feature_selector to use, by default "CatboostByShap"
        Possible values:
            - CatboostByShap
            - LAMA
    feature_selector_kwargs, optional
        Keyword arguments to initialize feature_selector, by default {}
        List of possible arguments and their default values:
            * CatboostByShap
                - n_features_to_select = 50
                - complexity = "Regular"
                - steps = 5
            * LAMA
                -task_type
                - target_colname
                - metric_name
                - metric_direction
                - timeout=120
                - model='lama'
                - strategy='RFA'
                - permutation_n_repeats = 5
    auto_models_init_kwargs, optional
        Keyword arguments to initialize AutoModel, by default {}
        List of possible arguments and their default values:
            - metric
            - time_series=False
            - models_list=None
            - blend=False
            - stack=False
    n_jobs, optional
        Number of cores for parallel computations, by default 1
    random_state, optional
        Random state, by default 42
    log_to_file, optional
        Whether to save logs in files.
        Save files locations:
            - ml_data/YYYY_mm_dd___HH-MM-SS/logs.log for info logs
            - ml_data/YYYY_mm_dd___HH-MM-SS/error.log for error, critical, warning logs
    """

    def __init__(
        self,
        task,
        use_preprocessing_pipeline=True,
        preprocessing_pipeline_kwargs={},
        use_val_test_pipeline=True,
        val_test_pipeline_kwargs={},
        feature_selector_type="CatboostByShap",
        feature_selector_kwargs={},
        auto_models_init_kwargs={},
        n_jobs=1,
        random_state=42,
        log_to_file=False,
    ):

        self.task = task
        self.random_state = random_state
        self.n_jobs = n_jobs

        if log_to_file:
            enable_logging_to_file()

        # initialize feature processings
        self.preprocessing_pipeline = None
        if use_preprocessing_pipeline:
            self.preprocessing_pipeline = PreprocessingPipeline(
                **preprocessing_pipeline_kwargs, random_state=self.random_state
            )

        self.val_test_pipeline = None
        if use_val_test_pipeline:
            self.val_test_pipeline = ValTestsPipeline(
                **val_test_pipeline_kwargs, random_state=self.random_state
            )

        self.feature_selector = None
        if feature_selector_type is not None:
            self.feature_selector = FEATURE_SELECTIONS_MAPPING[feature_selector_type](
                **feature_selector_kwargs,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        # initialize AutoModel
        self.auto_model = AutoModel(
            task=self.task,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **auto_models_init_kwargs,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        max_obs_for_preproc=100_000,
        auto_model_fit_kwargs={},
    ):
        """Fit the AutoML on train data.

        Parameters
        ----------
        X_train
            Pandas DataFrame with train features
        y_train
            Pandas DataFrame or numpy array of train target values
        X_test, optional
            Pandas DataFrame with test features, by default None
        y_test, optional
            Pandas DataFrame or numpy array of test target values, by default None
        max_obs_for_preproc, optional
            Maximum number of observations used to fit preprocessings, by default 100_000
            Smaller values speed up the execution, but may lead to inaccuracies.
        auto_model_fit_kwargs, optional
            Keyword arguments to fit AutoModel, by default {}
            List of possible arguments and their default values:
                - tuning_timeout=60
                - save_models=False
                - save_params=True
                - save_oof=False
                - save_test=False
        """

        # apply feature processings
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline.fit(
                X_train.iloc[-max_obs_for_preproc:], y_train[-max_obs_for_preproc:]
            )
            X_train = self.preprocessing_pipeline.transform(X_train)
            X_test = self.preprocessing_pipeline.transform(X_test)

        if self.val_test_pipeline is not None:
            self.val_test_pipeline.fit(
                X_train.iloc[-max_obs_for_preproc:], X_test.iloc[-max_obs_for_preproc:]
            )
            X_train = self.val_test_pipeline.transform(X_train)
            X_test = self.val_test_pipeline.transform(X_test)

        categorical_features = X_train.columns[
            (X_train.columns.str.startswith("OneHotEncoder"))
            | (X_train.columns.str.startswith("OrdinalEncoder"))
        ].tolist()

        if self.feature_selector is not None:
            self.feature_selector.fit(
                X_train.iloc[-max_obs_for_preproc:],
                y_train[-max_obs_for_preproc:],
                categorical_features=categorical_features,
            )
            X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)

            categorical_features = np.intersect1d(
                categorical_features, X_train.columns.tolist()
            ).tolist()

        # fit the AutoModel
        self.auto_model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            categorical_features=categorical_features,
            **auto_model_fit_kwargs,
        )

    def predict(self, X_test):
        """Inference the AutoML

        Parameters
        ----------
        X_test
            Pandas DataFrame with test features
        """

        # apply feature processings
        if self.preprocessing_pipeline is not None:
            X_test = self.preprocessing_pipeline.transform(X_test)

        if self.val_test_pipeline is not None:
            X_test = self.val_test_pipeline.transform(X_test)

        if self.feature_selector is not None:
            X_test = self.feature_selector.transform(X_test)

        # inference the AutoModel
        return self.auto_model.predict(X_test)
