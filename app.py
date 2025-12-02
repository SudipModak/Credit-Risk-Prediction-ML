import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("ExtraTreesClassifier_GermanCreditModel.pkl")

# Load encoders (ONLY the ones you have)
encoders = {
    "Sex": joblib.load("Sex_encoder.pkl"),
    "Housing": joblib.load("Housing_encoder.pkl"),
    "Purpose": joblib.load("Purpose_encoder.pkl")
}

st.title("Credit Risk Prediction App")

# ------------------- USER INPUT FIELDS -------------------

sex = st.selectbox("Sex", encoders["Sex"].classes_)
housing = st.selectbox("Housing", encoders["Housing"].classes_)
purpose = st.selectbox("Purpose", encoders["Purpose"].classes_)

credit_amount = st.number_input("Credit Amount", min_value=0)
duration = st.number_input("Duration (in months)", min_value=1)

# ------------------- PREPARE INPUT -----------------------

input_data = {
    'Sex': encoders["Sex"].transform([sex])[0],
    'Housing': encoders["Housing"].transform([housing])[0],
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': encoders["Purpose"].transform([purpose])[0]
}

input_df = pd.DataFrame([input_data])

# ------------------- PREDICTION ---------------------------

if st.button("Assess Risk"):
    pred = model.predict(input_df)[0]
    
    if pred == 1:
        st.success("The credit risk is LOW ✔")
    else:
        st.error("The credit risk is HIGH ❌")