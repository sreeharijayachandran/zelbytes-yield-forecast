import joblib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/interim/02_cleaned.parquet")
features = ["temperature", "humidity", "CO2", "yield"]
print(df.head())
scaler = joblib.load("models/minmax_scaler.joblib")

print(scaler.data_min_)
print(scaler)