import streamlit as st
import numpy as np
import joblib

# Model, feature selector ve scaler'ı yükle
model = joblib.load('diabetes_model.pkl')
sfm = joblib.load('feature_selector.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title("Diabetes Prediction")
st.write("Fill the following information to predict your diabetes risk:")

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)         # Ölçekleme
    input_selected = sfm.transform(input_scaled)         # Özellik seçimi
    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0][1]
    if prediction == 1:
        st.error(f"High diabetes risk! (Probability: %{probability*100:.2f})")
    else:
        st.success(f"Low diabetes risk. (Probability: %{probability*100:.2f})")
