import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the saved Keras model
from keras.models import load_model

keras_model = load_model("C:\\Users\\ayoba\\OneDrive - Ashesi University\\Year 2 sem 2 fall\\AI\\assignment 3\\final_keras_model1.h5")


# Load the scaler used during training
scaler = load('C:\\Users\\ayoba\\OneDrive - Ashesi University\\Year 2 sem 2 fall\\AI\\assignment 3\\scaler.joblib')

# Function to preprocess user inputs
def preprocess_input(user_input):
    # Assuming user_input is a dictionary with keys as feature names
    user_df = pd.DataFrame(user_input, index=[0])
    # Scale user input using the loaded scaler
    scaled_input = scaler.transform(user_df)
    return scaled_input

# Streamlit app
def main():
    st.title('Customer Churn Prediction App')
    st.write('Enter Customer Details')

    # Accept user inputs
    customer_data = {}
    customer_data['SeniorCitizen'] = st.slider('Senior Citizen (0 or 1)', 0, 1, 0)
    customer_data['tenure'] = st.slider('Tenure', 0, 72, 0)
    customer_data['MonthlyCharges'] = st.slider('Monthly Charges', 0, 200, 0)
    customer_data['TotalCharges'] = st.slider('Total Charges', 0, 8000, 0)
    # Add more input fields for other features
    customer_data['gender'] = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    customer_data['Partner'] = st.selectbox('Partner', ['Yes', 'No'])
    customer_data['Dependents'] = st.selectbox('Dependents', ['Yes', 'No'])
    customer_data['PhoneService'] = st.selectbox('Phone Service', ['Yes', 'No'])
    customer_data['MultipleLines'] = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    customer_data['InternetService'] = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    customer_data['OnlineSecurity'] = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    customer_data['OnlineBackup'] = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    customer_data['DeviceProtection'] = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    # Add more input fields for other features

    if st.button('Predict Churn'):
        # Preprocess user input
        processed_data = preprocess_input(customer_data)

        # Make predictions using the loaded model
        prediction = keras_model.predict(processed_data)

        # Display prediction result
        if prediction > 0.5:
            st.write('Churn: Yes')
        else:
            st.write('Churn: No')

if __name__ == '__main__':
    main()
