from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")
feature_cols = ["temperature", "humidity", "CO2"]
X_train = train[feature_cols]   
y_train = train["yield"]
X_test = test[feature_cols]
y_test = test["yield"]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"RF Test MAE:  {mae:.2f} kg")
print(f"RF Test RMSE: {rmse:.2f} kg")
print(f"RF Test R²:   {r2:.3f}")

importances = rf.feature_importances_
labels = ["temperature", "humidity", "co2"]
plt.barh(labels, importances)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("reports/figures/rf_importance.png", dpi=150)

joblib.dump(rf, "models/random_forest.joblib")