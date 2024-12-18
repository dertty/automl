import sys; sys.path.append("../")
import re
from pprint import pp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from src.automl import AutoML
import yaml
import joblib

with open("config.yaml", "r") as f:
    cfg = yaml.load(f, yaml.SafeLoader)

################################################################################################
#                           PREPARE DATA                                                       #
# Dataset: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction          #
################################################################################################
df = pd.read_csv("../data/airlines_train.csv").drop(columns="Unnamed: 0").sample(n=10_000, random_state=cfg["RANDOM_STATE"])

# intentionally add a constant column
df = df.assign(cnst=1)

X, y = df.drop(columns=cfg["TARGET"]), df[cfg["TARGET"]]

# rename X columns to remove "-" symbol (not processed by catboost)
X = X.rename(columns = lambda x:re.sub('-', '', x))
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test  = train_test_split(X, y, stratify=y, random_state=cfg["RANDOM_STATE"], test_size=0.2)
X_test, X_infernece, y_test, y_infernece  = train_test_split(X_test, y_test, stratify=y_test, random_state=cfg["RANDOM_STATE"], test_size=0.2)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_infernece = X_test.reset_index(drop=True)


################################################################################################
#                           INIT AUTOML                                                        #
################################################################################################
automl = AutoML(
    task=cfg["TASK"],
    preprocessing_pipeline_kwargs=cfg["preprocessing_pipeline_kwargs"],
    feature_selector_type=cfg["feature_selector_type"],
    feature_selector_kwargs=cfg["feature_selector_kwargs"],
    auto_models_init_kwargs=cfg["auto_models_init_kwargs"],
    n_jobs=cfg["N_JOBS"],
    log_to_file=cfg["LOG_TO_FILE"]
)


################################################################################################
#                           FIT AUTOML                                                         #
################################################################################################
automl.fit(X_train, y_train, X_test, y_test, auto_models_fit_kwargs=cfg["auto_models_fit_kwargs"])


################################################################################################
#                           SAVE AUTOML                                                        #
################################################################################################
joblib.dump(automl, "automl.joblib")


################################################################################################
#                           LOAD SAVED AND INFERENCE                                                        #
################################################################################################
automl = joblib.load("automl.joblib")
automl.predict(X_test)
