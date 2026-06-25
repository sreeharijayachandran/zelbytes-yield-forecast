# Monitoring Plan

## Sample Prediction Log

| Timestamp  | Temperature | Humidity | CO₂     | Predicted Yield |
| ---------- | ----------- | -------- | ------- | --------------- |
| 2026-06-24 | 25.0°C      | 88%      | 800 ppm | 1.20 kg         |

## Retraining Triggers

The model should be retrained when:

* New sensor data becomes available.
* Prediction accuracy decreases significantly.
* Growing conditions change substantially.
* MAE increases above acceptable limits.

## Model Artifact Handling

The deployed application loads:

* random_forest_tuned.joblib
* minmax_scaler_train.joblib

Artifacts are stored in the repository under the `models/` directory and loaded during application startup.
