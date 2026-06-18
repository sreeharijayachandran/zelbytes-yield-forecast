import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_yield

st.set_page_config(
page_title="Zelbytes Agritech",
page_icon="🍄",
layout="wide"
)

st.title("🍄 Mushroom Yield Forecast")
st.caption("Zelbytes Agritech Decision Support Tool")

st.markdown(
"""
Enter current polyhouse sensor readings to estimate mushroom yield.
"""
)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)",value=25.0)


    humidity = st.number_input("Humidity (%)",value=88.0)

co2 = st.number_input(
    "CO₂ (ppm)",
    value=900.0
)


with col2:


 if temperature < 18 or temperature > 30:
    st.warning("Temperature is outside the typical training range.")

 if humidity < 75 or humidity > 95:
    st.warning("Humidity is outside the typical training range.")

 if co2 < 500 or co2 > 1200:
    st.warning("CO₂ is outside the typical training range.")

prediction=0 

if st.button("Predict Yield"):

 prediction = predict_yield(
    temperature,
    humidity,co2)

st.metric(
    "Predicted Yield",
    f"{prediction:.2f} kg"
)

st.subheader("Yield Sensitivity to Humidity")

humidity_range = np.linspace(75, 95, 25)

yields = [
    predict_yield(
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

st.line_chart(
    chart_df,
    x="Humidity",
    y="Predicted Yield"
)
