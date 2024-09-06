import numpy as np
import pickle as pkl
import streamlit as st
import plotly.express as px

# Load the classifier and scalers
with open('.pkl/classifier.pkl', 'rb') as pickle_in:
    classifier = pkl.load(pickle_in)

with open('.pkl/mms_scaler.pkl', 'rb') as mms_scaler_file:
    mms_scaler = pkl.load(mms_scaler_file)

with open('.pkl/std_scaler.pkl', 'rb') as std_scaler_file:
    std_scaler = pkl.load(std_scaler_file)

# Prediction function
def prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    # Mappings for categorical features
    sex_mapping = {'Male': 0, 'Female': 1}
    chest_pain_mapping = {'Atypical Angina': 0, 'Non-Anginal Pain': 1, 'Asymptomatic': 2, 'Typical Angina': 3}
    exercise_angina_mapping = {'No': 0, 'Yes': 1}
    st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

    # Convert categorical inputs to numerical values
    Sex = sex_mapping[Sex]
    ChestPainType = chest_pain_mapping[ChestPainType]
    ExerciseAngina = exercise_angina_mapping[ExerciseAngina]
    ST_Slope = st_slope_mapping[ST_Slope]

    # Prepare arrays for features
    MaxHR_scaled = mms_scaler.transform(np.array([[MaxHR]]))  # Scale MaxHR using MinMaxScaler
    Age_scaled = mms_scaler.transform(np.array([[Age]]))      # Scale Age using MinMaxScaler
    Cholesterol_scaled = std_scaler.transform(np.array([[Cholesterol]]))  # Scale Cholesterol using StandardScaler
    Oldpeak_scaled = std_scaler.transform(np.array([[Oldpeak]]))          # Scale Oldpeak using StandardScaler
    RestingBP_scaled = std_scaler.transform(np.array([[RestingBP]]))      # Scale RestingBP using StandardScaler

    # Concatenate scaled features and other features
    input_data = np.concatenate((
        MaxHR_scaled, Age_scaled,   # MinMax scaled features
        Cholesterol_scaled, Oldpeak_scaled, RestingBP_scaled,  # Standard scaled features
        np.array([[Sex, ChestPainType, ExerciseAngina, ST_Slope]])  # Other features
    ), axis=1)

    # Make a prediction
    prediction = classifier.predict(input_data)  # Predicted class (0 or 1)
    probability = classifier.predict_proba(input_data)  # Predicted probabilities for each class
    return prediction, probability

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Heart Failure Prediction", page_icon="🫀", layout="centered")
    st.header('🫀 Heart Disease Prediction App')
    st.write("Please enter the patient details and click 'Predict' to get the prediction.")

    # Input form for prediction
    with st.form(key='prediction_form'):
        Age = st.number_input("Age", min_value=0, max_value=200, value=0)
        Sex = st.selectbox("Sex", ["Male", "Female"])
        ChestPainType = st.selectbox("Chest Pain Type", ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"])
        RestingBP = st.number_input("Resting Blood Pressure [mm Hg]", min_value=0, max_value=200, value=0)
        Cholesterol = st.number_input("Serum Cholesterol [mm/dl]", min_value=0, max_value=500, value=0)
        MaxHR = st.number_input("Maximum Heart Rate", min_value=56, max_value=208, value=56)
        ExerciseAngina = st.selectbox("Exercise Angina", ["No", "Yes"])
        Oldpeak = st.number_input("Oldpeak [Numeric value measured in depression]", min_value=-5.0, max_value=7.0, value=0.0)
        ST_Slope = st.selectbox("The slope of the peak exercise ST segment", ["Up", "Flat", "Down"])

        # Predict button
        submitted = st.form_submit_button("Predict")

    if submitted:
        result, probability = prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)
        if result[0] == 0:
            st.success("Prediction: Normal (No Heart Disease)")
        else:
            st.error("Prediction: Have Heart Failure")
        
        st.write(f"Probability of Normal: {probability[0][0] * 100:.2f}%")
        st.write(f"Probability of having Heart Failure: {probability[0][1] * 100:.2f}%")

    st.info('Copyright © Ardava Barus - All rights reserved')

if __name__ == '__main__':
    main()
