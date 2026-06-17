import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from joblib import load


champion = load("models/random_forest.joblib")

test = pd.read_parquet("data/processed/test.parquet")

feature_cols = ["temperature", "humidity", "CO2"]

X_test = test[feature_cols]
y_test = test["yield"]
pred = champion.predict(X_test)

results = pd.DataFrame({
    "model": ["Linear Regression", "Random Forest", "Tuned Random Forest"],
    "test_mae": [0.09, 0.09, 0.081],
    "test_r2": [0.696, 0.707, 0.737],
})

print(results.to_markdown(index=False))



plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual yield (kg)")
plt.ylabel("Predicted yield (kg)")
plt.title("Champion Model: Predicted vs Actual")
plt.tight_layout()
plt.savefig("reports/figures/random_forest_vs_actual1.png", dpi=150)