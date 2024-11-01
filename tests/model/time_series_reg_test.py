import sys; sys.path.append("../")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder

from src.ml_models import AutoML
from src.ml_models.metrics import MSE


df = pd.read_csv("../data/time_series/time_series_reg")

df = df.sort_values(by="date_time").reset_index(drop=True)
df_train = df.loc[df.date_time < "2018-01-10 15:36:01"].drop(columns="date_time")
df_test = df.loc[df.date_time >= "2018-01-10 15:36:01"].drop(columns="date_time")

X_train, X_test = df_train.drop(columns="Room_Occupancy_Count"), df_test.drop(columns="Room_Occupancy_Count")
y_train, y_test = df_train["Room_Occupancy_Count"], df_test["Room_Occupancy_Count"]

model = AutoML(task="regression", n_jobs=7, metric=MSE(), tuning_timeout=10, time_series=False)
model.fit(X_train, y_train, X_test, y_test)
print(model.predict(X_test)[:4])
print(model.best_model.name)