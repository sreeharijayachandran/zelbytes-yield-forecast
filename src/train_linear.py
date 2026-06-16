from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")
feature_cols= ["temperature", "humidity", "CO2"]
X_train = train[feature_cols]
y_train = train["yield"]

X_test = test[feature_cols]
y_test = test["yield"]

model = LinearRegression()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"Test MAE:  {mae:.2f} kg")
print(f"Test RMSE: {rmse:.2f} kg")
print(f"Test R²:   {r2:.3f}")

for name, coef in zip(["temp", "humidity", "co2"], model.coef_):
    print(f"  coef {name}: {coef:.3f}")

joblib.dump(model, "models/linear_regression.joblib")