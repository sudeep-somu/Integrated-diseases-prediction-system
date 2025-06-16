import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import zscore
import pandas as pd

# Load unified model
model = load_model(r'C:\Users\sudee\Desktop\minor2project\parkinsons_model.h5')

# Page Setup
st.set_page_config(page_title="Multi-Disease Prediction", layout="centered")
st.title("🧬 CNN-Based Comprehensive Disease Prediction System")

# Disease selection
option = st.sidebar.selectbox("Choose Prediction Type", [
    "🧠 Parkinson's",
    "🩸 Diabetes",
    "⚖️ Adiposity Risk"
])

# Feature Inputs
def get_scaled_input(values):
    df = pd.DataFrame([values])
    return df.apply(zscore, axis=1)

if option == "🧠 Parkinson's":
    st.subheader("Enter features for Parkinson's prediction:")
    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
        'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    inputs = [st.number_input(f, key=f) for f in features]
    if st.button("Predict Parkinson's"):
        scaled = get_scaled_input(inputs)
        preds = model.predict(scaled)
        result = preds[0][0] if isinstance(preds, list) else preds[0][0]
        st.success("🧠 Positive" if result > 0.5 else "✅ Negative")

elif option == "🩸 Diabetes":
    st.subheader("Enter features for Diabetes prediction:")
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]
    inputs = [st.number_input(f, key=f) for f in features]
    if st.button("Predict Diabetes"):
        # Padding to match input dimension
        full_input = [0] * 22 + inputs + [0] * 5  # Parkinson(22) + Diabetes(8) + Adiposity(5)
        scaled = get_scaled_input(full_input)
        preds = model.predict(scaled)
        result = preds[1][0] if isinstance(preds, list) else preds[0][1]
        st.success("🩸 Positive" if result > 0.5 else "✅ Negative")

elif option == "⚖️ Adiposity Risk":
    st.subheader("Enter features for Adiposity prediction:")
    features = ['Height', 'Weight', 'Waist', 'Hip', 'Age']
    inputs = [st.number_input(f, key=f) for f in features]
    if st.button("Predict Adiposity Risk"):
        # Padding to match input dimension
        full_input = [0] * 30 + inputs  # Parkinson(22) + Diabetes(8) + Adiposity(5)
        scaled = get_scaled_input(full_input)
        preds = model.predict(scaled)
        result = preds[2][0] if isinstance(preds, list) else preds[0][2]
        st.success("⚠️ High Risk" if result > 0.5 else "✅ Low Risk")
