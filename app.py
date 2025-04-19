import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 🔹 Load model artifacts
model = joblib.load("financial_inclusion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("💳 Financial Inclusion Predictor")
st.write("Fill the form below to predict if someone is likely to have a bank account.")

# 🔹 Create input form based on your dataset features
user_inputs = {}

# These fields must match your dataset — adjust as needed
for feature in feature_names:
    if feature in label_encoders:
        user_inputs[feature] = st.selectbox(f"{feature}", label_encoders[feature].classes_)
    else:
        user_inputs[feature] = st.number_input(f"{feature}", step=1.0)

# 🔹 Predict button
if st.button("🔍 Predict"):
    # Encode categorical features
    for col in user_inputs:
        if col in label_encoders:
            user_inputs[col] = label_encoders[col].transform([user_inputs[col]])[0]

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_inputs], columns=feature_names)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    result = "✅ Likely to Have a Bank Account" if prediction == 1 else "❌ Unlikely to Have a Bank Account"
    st.success(result)
