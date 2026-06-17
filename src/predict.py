import json
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

_scaler = joblib.load(MODEL_DIR / "minmax_scaler_train.joblib")
_model = joblib.load(MODEL_DIR / "random_forest_tuned.joblib")

_feature_cols = ["temperature", "humidity", "CO2"]

def predict_yield(temperature_c: float,
                  humidity_pct: float,
                  co2_ppm: float) -> float:

    row = pd.DataFrame(
        [[temperature_c, humidity_pct, co2_ppm]],
        columns=_feature_cols
    )

    print("Input:", row)

    return float(_model.predict(row)[0])


if __name__ == "__main__":
    print(predict_yield(25.6,84.7,772.0))
    print(predict_yield(23.3,91.5,811.0))