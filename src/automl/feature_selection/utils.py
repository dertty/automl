import pandas as pd
import numpy as np

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)

def find_correlated_features(
    X: pd.DataFrame,
    variables,
    method: str,
    threshold: float,
):
    """
    Much faster of way of computing correlation.
    Uses `np.corrcoef` inside. For now, only `spearman` correlation is available.
    """
    # the correlation matrix
    correlated_matrix = np.corrcoef(X[variables].to_numpy(), rowvar=False)

    # the correlated pairs
    correlated_mask = np.triu(np.abs(correlated_matrix), 1) > threshold

    examined = set()
    correlated_groups = list()
    features_to_drop = list()
    correlated_dict = {}
    for i, f_i in enumerate(variables):
        if f_i not in examined:
            examined.add(f_i)
            temp_set = set([f_i])
            for j, f_j in enumerate(variables):
                if f_j not in examined:
                    if correlated_mask[i, j] == 1:
                        examined.add(f_j)
                        features_to_drop.append(f_j)
                        temp_set.add(f_j)
            if len(temp_set) > 1:
                correlated_groups.append(temp_set)
                correlated_dict[f_i] = temp_set.difference({f_i})

    return correlated_groups, features_to_drop, correlated_dict


class SmartCorrelatedSelectionFast(BaseSelector):

    """
    Much faster version of `feature_engine.selection.SmartCorrelatedSelection`.
    Uses faster version of `find_correlated_features` function.
    """

    def __init__(
        self,
        variables=None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        selection_method: str = "missing_values",
        estimator=None,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        confirm_variables: bool = False,
    ):
        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError(
                f"`threshold` must be a float between 0 and 1. Got {threshold} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if selection_method not in [
            "missing_values",
            "cardinality",
            "variance",
            "model_performance",
        ]:
            raise ValueError(
                "selection_method takes only values 'missing_values', 'cardinality', "
                f"'variance' or 'model_performance'. Got {selection_method} instead."
            )

        if selection_method == "model_performance" and estimator is None:
            raise ValueError(
                "Please provide an estimator, e.g., "
                "RandomForestClassifier or select another "
                "selection_method."
            )

        if selection_method == "missing_values" and missing_values == "raise":
            raise ValueError(
                "When `selection_method = 'missing_values'`, you need to set "
                f"`missing_values` to `'ignore'`. Got {missing_values} instead."
            )

        super().__init__(confirm_variables)

        self.variables = variables
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values
        self.selection_method = selection_method
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        # check input dataframe
        X = check_X(X)

        self.variables_ = X.select_dtypes(include="number").columns.tolist()

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        if self.selection_method == "model_performance" and y is None:
            raise ValueError(
                "When `selection_method = 'model_performance'` y is needed to "
                "fit the transformer."
            )

        if self.selection_method == "missing_values":
            features = (
                X[self.variables_]
                .isnull()
                .sum()
                .sort_values(ascending=True)
                .index.to_list()
            )
        elif self.selection_method == "variance":
            features = (
                X[self.variables_].std().sort_values(ascending=False).index.to_list()
            )
        elif self.selection_method == "cardinality":
            features = (
                X[self.variables_]
                .nunique()
                .sort_values(ascending=False)
                .index.to_list()
            )
        else:
            features = sorted(self.variables_)

        correlated_groups, features_to_drop, correlated_dict = find_correlated_features(
            X, features, self.method, self.threshold
        )

        # select best performing feature according to estimator
        if self.selection_method == "model_performance":
            correlated_dict = dict()
            cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv
            for feature_group in correlated_groups:
                feature_performance, _ = single_feature_performance(
                    X=X,
                    y=y,
                    variables=feature_group,
                    estimator=self.estimator,
                    cv=cv,
                    groups=self.groups,
                    scoring=self.scoring,
                )
                # get most important feature
                f_i = (
                    pd.Series(feature_performance).sort_values(ascending=False).index[0]
                )
                correlated_dict[f_i] = feature_group.difference({f_i})

            # convoluted way to pick up the variables from the sets in the
            # order shown in the dictionary. Helps make transformer deterministic
            features_to_drop = [
                variable
                for set_ in correlated_dict.values()
                for variable in sorted(set_)
            ]

        self.features_to_drop_ = features_to_drop
        self.correlated_feature_sets_ = correlated_groups
        self.correlated_feature_dict_ = correlated_dict

        # save input features
        self._get_feature_names_in(X)

        return self