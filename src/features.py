import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
import joblib
df=pd.read_parquet("data/interim/02_cleaned.parquet").sort_values("timestamp")
df["temp_humid_interaction"] = df["temperature"] * df["humidity"]/100
features = ["temperature", "humidity", "CO2", "temp_humid_interaction"]
x=df[features]
y=df["yield"]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, "models/minmax_scaler.joblib")
processed=pd.DataFrame(x_scaled, columns=[c + "_scaled" for c in features])
processed["yield"] = y.values
processed.to_parquet("data/processed/features.parquet", index=False)