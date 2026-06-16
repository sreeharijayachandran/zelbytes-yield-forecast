from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
feature_cols = ["temperature", "humidity", "CO2"]
X_train = train[feature_cols]
y_train = train["yield"]


tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
lin = LinearRegression()

rf_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
lin_scores = cross_val_score(lin, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

print("RF CV MAE:", (-rf_scores).mean(), "+/-", (-rf_scores).std())
print("Linear CV MAE:", (-lin_scores).mean(), "+/-", (-lin_scores).std())