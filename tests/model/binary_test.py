import sys; sys.path.append("../../")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder

from src.automl.model import AutoML
from src.automl.model.metrics import Accuracy, RocAuc


df = pd.read_csv("../../data/classification/data.csv", sep=";")

X = df.drop(columns="Target")
y = df["Target"]
# transform to binary problem
y[y=="Graduate"] = "Enrolled"
y = pd.Series(LabelEncoder().fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)







model = AutoML(task="classification", n_jobs=7, metric=Accuracy(), tuning_timeout=30)
model.fit(X_train, y_train, X_test, y_test)
print(model.predict(X_test)[:4])
print(model.best_model.name)