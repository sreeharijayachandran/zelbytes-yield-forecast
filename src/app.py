import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

try:
    from predict import predict_yield

    @st.cache_resource
    def load_model():
        return load("models/random_forest_tuned.joblib")

    model = load_model()

except FileNotFoundError:
    st.error(
        "Model file not found. Please verify deployment artifacts."
    )
    st.stop()

@st.cache_data
def cached_predict(temperature, humidity, co2):
    return predict_yield(temperature, humidity, co2)

st.set_page_config(
page_title="Zelbytes Agritech",
page_icon="🍄",
layout="wide"
)

st.title("🍄 Mushroom Yield Forecast")

st.markdown(
"""
Enter current polyhouse sensor readings to estimate mushroom yield.
"""
)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)",value=25.0)

    humidity = st.number_input("Humidity (%)",value=88.0)

    co2 = st.number_input("CO₂ (ppm)",value=800.0)


with col2:


 if temperature < 18 or temperature > 30:
    st.warning("Temperature is outside the typical training range.")

 if humidity < 75 or humidity > 95:
    st.warning("Humidity is outside the typical training range.")

 if co2 < 500 or co2 > 1000:
    st.warning("CO₂ is outside the typical training range.")

if st.button("Predict Yield"):

    with st.spinner("Generating prediction..."):
        prediction = cached_predict(
            temperature,
            humidity,
            co2
        )

    st.metric(
        "Predicted Yield",
        f"{prediction:.2f} kg"
    )

col3, col4, col5,col6= st.columns(4)
with col3:
    
    st.subheader("Yield Sensitivity to CO₂")

    co2_range = np.linspace(600, 900,30)

    yields = [
        cached_predict(
            temperature,
            humidity,
            c
        )
        for c in co2_range
    ]

    chart_df = pd.DataFrame({
        "CO₂": co2_range,
        "Predicted Yield": yields
    })

    st.line_chart(
        chart_df,
        x="CO₂",
        y="Predicted Yield"
    )
with col4:
    st.subheader("Yield Sensitivity to Humidity")

    humidity_range = np.linspace(75, 95, 25)

    yields = [
     cached_predict(
        temperature,
        h,
        co2
    )
    for h in humidity_range
    ]

    chart_df = pd.DataFrame({
    "Humidity": humidity_range,
    "Predicted Yield": yields
    })

    st.line_chart(chart_df,x="Humidity",y="Predicted Yield")
with col5:
    st.subheader("Yield Sensitivity to Temperature")

    temperature_range = np.linspace(21,28, 25)

    yields = [
     cached_predict(
        t,
        humidity,
        co2
     )
     for t in temperature_range
    ]

    chart_df = pd.DataFrame({
    "Temperature": temperature_range,
    "Predicted Yield": yields
     })

    st.line_chart(chart_df, x="Temperature", y="Predicted Yield")
with col6:
   st.subheader("Feature Contribution to Yield Prediction")
   
   importances = model.feature_importances_

   fig, ax = plt.subplots(figsize=(6, 6),facecolor="none")

   ax.set_facecolor("none")

   ax.pie(importances,labels=["Temperature", "Humidity", "CO₂"],autopct="%1.1f%%",startangle=90,textprops={"color": "white"})

   ax.set_title("Feature Contribution to Yield Prediction")

   st.pyplot(fig)