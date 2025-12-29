import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("concrete_model.pkl")

st.title("Concrete Compressive Strength Prediction")

st.write("Enter the concrete mix properties below:")

cement = st.number_input("Cement (kg/m³)", min_value=0.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0)
coarse = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0)
fine = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0)
age = st.number_input("Age (days)", min_value=1)

if st.button("Predict Strength"):
    input_data = np.array([[cement, slag, flyash, water,
                             superplasticizer, coarse, fine, age]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")