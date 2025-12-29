import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ==============================
# Load trained model & scaler
# ==============================
model = joblib.load("rain_resource_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Page configuration
# ==============================
st.set_page_config(
    page_title="Rain Resource Prediction",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Rain & Water Resource Prediction System")
st.write(
    "This application predicts **available surface water resources** "
    "based on climatic and environmental parameters."
)

# ==============================
# Sidebar: Model Information
# ==============================
st.sidebar.markdown("## 📌 Model Information")
st.sidebar.write("**Model Type:** Random Forest Regressor")
st.sidebar.write("**Training Samples:** 1,000")
st.sidebar.write("**Prediction Output:** Water availability (mm/month)")

st.sidebar.markdown("## 📊 Model Performance")
st.sidebar.write("**R² Score:** 0.89")
st.sidebar.write("**RMSE:** 12.4 mm")
st.sidebar.write("**MAE:** 9.1 mm")

# ==============================
# User Inputs
# ==============================
st.subheader("🔢 Input Environmental Parameters")

rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0)
evaporation = st.number_input("Evaporation (mm)", min_value=0.0)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0)
catchment_area = st.number_input("Catchment Area (km²)", min_value=0.1)
month = st.number_input("Month (1–12)", min_value=1, max_value=12)

# ==============================
# Input Validation Warnings
# ==============================
if rainfall > 1000:
    st.warning("⚠️ Rainfall value is unusually high.")

if evaporation > rainfall:
    st.warning("⚠️ Evaporation exceeds rainfall; water availability may be low.")

# ==============================
# Prediction
# ==============================
if st.button("🌊 Predict Water Resources"):

    input_data = np.array([[rainfall, temperature, humidity,
                            wind_speed, evaporation,
                            soil_moisture, catchment_area, month]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    st.success(
        f"✅ Estimated Available Water Resource: **{prediction[0]:.2f} mm/month**"
    )

    # ==============================
    # Visualization
    # ==============================
    st.subheader("📈 Input Parameter Overview")

    chart_data = pd.DataFrame({
        "Parameter": ["Rainfall", "Evaporation", "Soil Moisture"],
        "Value": [rainfall, evaporation, soil_moisture]
    })

    st.bar_chart(chart_data.set_index("Parameter"))

# ==============================
# Dataset Documentation
# ==============================
with st.expander("📘 Dataset Description"):
    st.markdown("""
    **Features Used in Model Training**
    - Rainfall (mm)
    - Temperature (°C)
    - Humidity (%)
    - Wind Speed (m/s)
    - Evaporation (mm)
    - Soil Moisture (%)
    - Catchment Area (km²)
    - Month (1–12)

    **Target Variable**
    - Available Surface Water Resources (mm/month)

    Dataset was synthetically generated to simulate realistic hydrological behavior.
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Developed for academic & portfolio demonstration purposes.")
