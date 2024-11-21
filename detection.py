# detection.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model
model = joblib.load('models/heart_disease_model.pkl')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction App",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App title and description
st.title('❤️ Heart Disease Risk Prediction App')
st.write("""
This app predicts the **risk of heart disease** based on personal health data.
Please provide the following information:
""")

# Function to get user input
def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
    chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment (slope)', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', options=[0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('Your Input Data')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display results
st.subheader('Risk Assessment')
risk = 'High Risk' if prediction[0] == 1 else 'Low Risk'
st.write(f'**{risk}** of heart disease.')

st.subheader('Prediction Probability')
st.write(f'Probability of No Heart Disease: **{prediction_proba[0][0]:.2%}**')
st.write(f'Probability of Heart Disease: **{prediction_proba[0][1]:.2%}**')

# Disclaimer
st.write("""
---
**Disclaimer:** This app is for educational purposes only and should not be used as a substitute for professional medical advice. Consult with a qualified healthcare provider for any medical concerns.
""")