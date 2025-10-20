import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# UI Title
st.title("Customer Churn Prediction App")

# Input Fields (CustomerID Removed)
age = st.number_input("Age", min_value=18, max_value=120, step=1, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (Months)", min_value=0, step=1, value=12)
usage_frequency = st.number_input("Usage Frequency", min_value=0, step=1, value=5)
support_calls = st.number_input("Support Calls", min_value=0, step=1, value=1)
payment_delay = st.number_input("Payment Delay (Days)", min_value=0, step=1, value=0)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.number_input("Contract Length", min_value=1, step=1, value=12)
total_spend = st.number_input("Total Spend", min_value=0.0, step=1.0, value=100.0)
last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, step=1, value=5)

# Encode values manually
gender = 1 if gender == "Male" else 0
subscription_map = {"Basic": 0, "Standard": 1, "Premium": 2}
subscription_type = subscription_map[subscription_type]

# Prepare DataFrame (CustomerID Removed)
data = pd.DataFrame([[
    age, gender, tenure, usage_frequency,
    support_calls, payment_delay, subscription_type,
    contract_length, total_spend, last_interaction
]], columns=[
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Last Interaction"
])

# Predict Button
if st.button("Predict"):
    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        st.success(f"Customer Likely to Churn — Confidence: {confidence:.2f}%")
    else:
        st.error(f"Customer Not Likely to Churn — Confidence: {confidence:.2f}%")
