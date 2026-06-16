import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

feature_cols = ["temperature", "humidity", "CO2"]

X_test = test[feature_cols]
y_test = test["yield"]

model = joblib.load("models/linear_regression.joblib")

pred_test = model.predict(X_test)

residuals = y_test - pred_test

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(pred_test, residuals, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set(xlabel="Predicted yield (kg)", ylabel="Residual (kg)")

axes[1].scatter(X_test["humidity"],residuals,alpha=0.5)  # humidity feature column
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set(xlabel="Scaled humidity", ylabel="Residual (kg)")

plt.tight_layout()
plt.savefig("reports/figures/residuals_linear.png", dpi=150)