import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("weather_model.h5")

st.title("Weather-Based Ventilation Control")
st.write("Enter the last 7 temperature max values to predict the next one.")

# Input fields for the last 10 temp_max values
temp_inputs = []
for i in range(7):
    val = st.number_input(f"Temperature (°C) - Day {i+1}", min_value=-50.0, max_value=50.0, value=15.0)
    temp_inputs.append(val)

if st.button("Predict"):
    # Prepare input
    X_input = np.array(temp_inputs).reshape(1, 7, 1)

    # Prediction
    prediction = model.predict(X_input)
    predicted_temp = prediction[0][0]

    st.subheader(f"Predicted Temperature (Next Day): {predicted_temp:.2f}°C")

    # Ventilation logic
    if predicted_temp > 25:
        st.success("Ventilation Status: OPEN - It's going to be warm.")
    else:
        st.info("Ventilation Status: CLOSED - Cool or Moderate weather.")
