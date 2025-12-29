import streamlit as st
import joblib
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

st.title("My ML Model")

# Example inputs (change to match your model)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2]])
    st.success(f"Prediction: {prediction[0]}")
