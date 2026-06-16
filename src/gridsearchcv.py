import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

feature_cols = ["temperature", "humidity", "CO2"]

X_train = train[feature_cols]
y_train = train["yield"]

X_test = test[feature_cols]
y_test = test["yield"]

# Time-series cross validation
tscv = TimeSeriesSplit(n_splits=3)

# Parameter grid

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5],
}


rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
)


grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV MAE:", -grid.best_score_)

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# Save model
joblib.dump(
    best_model,
    "models/random_forest_tuned.joblib"
)

# Save parameters
with open("models/best_params.json", "w") as f:
    json.dump(grid.best_params_, f, indent=4)