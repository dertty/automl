import sys; sys.path.append("../")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

from src.ml_models import AutoML
from src.ml_models.metrics import MSE, MAE, MAPE


df = pd.read_csv("../data/regression/wine+quality/data.csv")

X = df.drop(columns="quality")
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = AutoML(task="regression", n_jobs=7, metric=MSE(), tuning_timeout=30)
model.fit(X_train, y_train, X_test, y_test)
print(model.predict(X_test))
