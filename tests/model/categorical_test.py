import sys; sys.path.append("../")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder

from src.ml_models import AutoML
from src.ml_models.metrics import Accuracy, RocAuc

CAT_COLUMN = "Application mode"
df = pd.read_csv("../data/classification/data.csv", sep=";")

X = df.drop(columns="Target")
X[CAT_COLUMN] = LabelEncoder().fit_transform(X[CAT_COLUMN])
y = df["Target"]
# transform to binary problem
y[y=="Graduate"] = "Enrolled"
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
model = AutoML(task="classification", n_jobs=7, metric=Accuracy(), tuning_timeout=10)
model.fit(X_train, y_train, X_test, y_test, categorical_features=[CAT_COLUMN])
print(model.predict(X_test)[:4])
print(model.best_model.name)