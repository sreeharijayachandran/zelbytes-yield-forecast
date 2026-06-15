import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_parquet("data/interim/02_cleaned.parquet").sort_values("timestamp")
feature = ["temperature", "humidity", "CO2"]

split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[feature])
X_test = scaler.transform(test[feature])
y_train = train["yield"].values
y_test = test["yield"].values

joblib.dump(scaler, "models/minmax_scaler_train.joblib")

print(f"Train: {train['timestamp'].min()} → {train['timestamp'].max()}")
print(f"Test:  {test['timestamp'].min()} → {test['timestamp'].max()}")

train.to_parquet(
    "data/processed/train.parquet",
    index=False
)

test.to_parquet(
    "data/processed/test.parquet",
    index=False
)