import csv
from datetime import datetime, timezone
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

_scaler = joblib.load(MODEL_DIR / "minmax_scaler_train.joblib")
_model = joblib.load(MODEL_DIR / "random_forest_tuned.joblib")

_feature_cols = ["temperature", "humidity", "CO2"]

# Log file location
LOG_PATH = Path("logs/predictions.csv")


def log_prediction(temp, humid, co2, predicted_kg):
    LOG_PATH.parent.mkdir(exist_ok=True)

    write_header = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "timestamp_utc",
                "temp_c",
                "humidity_pct",
                "co2_ppm",
                "predicted_kg"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            temp,
            humid,
            co2,
            round(predicted_kg, 3)
        ])


def predict_yield(
    temperature_c: float,
    humidity_pct: float,
    co2_ppm: float
) -> float:

    row = pd.DataFrame(
        [[temperature_c, humidity_pct, co2_ppm]],
        columns=_feature_cols
    )

    prediction = float(_model.predict(row)[0])

    # Save prediction to CSV log
    log_prediction(
        temperature_c,
        humidity_pct,
        co2_ppm,
        prediction
    )

    return prediction


if __name__ == "__main__":
    print(predict_yield(25.6, 84.7, 772.0))
    print(predict_yield(23.3, 91.5, 811.0))