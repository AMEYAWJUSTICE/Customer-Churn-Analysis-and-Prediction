import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained Logistic Regression model
try:
    log_reg_model = joblib.load('logistic_regression_model.joblib')
    st.success("Logistic Regression model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}. Please ensure 'logistic_regression_model.joblib' is in the same directory.")
    st.stop()

# Define the list of selected features used during training
selected_features = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen',
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

st.title("Telco Customer Churn Prediction")
st.write("Predict if a customer will churn based on their service attributes.")

st.markdown("### Customer Information")

# Create input widgets for each feature
input_data = {}

# Numeric features
input_data['tenure'] = st.slider('Tenure (months)', 0, 72, 12)
input_data['MonthlyCharges'] = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=50.0, step=0.1)
input_data['TotalCharges'] = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=500.0, step=0.1)

# Binary features (0/1)
input_data['SeniorCitizen'] = st.selectbox('Senior Citizen', ['No', 'Yes'], index=0)
input_data['gender'] = st.selectbox('Gender', ['Female', 'Male'], index=0)
input_data['Partner'] = st.selectbox('Partner', ['No', 'Yes'], index=0)
input_data['Dependents'] = st.selectbox('Dependents', ['No', 'Yes'], index=0)
input_data['PhoneService'] = st.selectbox('Phone Service', ['No', 'Yes'], index=1)
input_data['PaperlessBilling'] = st.selectbox('Paperless Billing', ['No', 'Yes'], index=1)

# Map binary inputs to 0/1
input_data['SeniorCitizen'] = 1 if input_data['SeniorCitizen'] == 'Yes' else 0
input_data['gender'] = 1 if input_data['gender'] == 'Male' else 0 # Female:0, Male:1
input_data['Partner'] = 1 if input_data['Partner'] == 'Yes' else 0
input_data['Dependents'] = 1 if input_data['Dependents'] == 'Yes' else 0
input_data['PhoneService'] = 1 if input_data['PhoneService'] == 'Yes' else 0
input_data['PaperlessBilling'] = 1 if input_data['PaperlessBilling'] == 'Yes' else 0

st.markdown("### Service Information")

# Handle one-hot encoded features by grouping them logically

# MultipleLines
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'], index=0)
input_data['MultipleLines_No phone service'] = (multiple_lines == 'No phone service')
input_data['MultipleLines_Yes'] = (multiple_lines == 'Yes')

# InternetService
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], index=0)
input_data['InternetService_Fiber optic'] = (internet_service == 'Fiber optic')
input_data['InternetService_No'] = (internet_service == 'No')

# Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
# These were encoded with drop_first=True, so only the '_Yes' column exists in selected_features.
# 'No internet service' and 'No' both map to False for the '_Yes' column.

def get_service_status(service_name):
    option = st.selectbox(service_name, ['No', 'Yes', 'No internet service'], index=0)
    return (option == 'Yes')

input_data['OnlineSecurity_Yes'] = get_service_status('Online Security')
input_data['OnlineBackup_Yes'] = get_service_status('Online Backup')
input_data['DeviceProtection_Yes'] = get_service_status('Device Protection')
input_data['TechSupport_Yes'] = get_service_status('Tech Support')
input_data['StreamingTV_Yes'] = get_service_status('Streaming TV')
input_data['StreamingMovies_Yes'] = get_service_status('Streaming Movies')

# Contract
contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'], index=0)
input_data['Contract_One year'] = (contract_type == 'One year')
input_data['Contract_Two year'] = (contract_type == 'Two year')

# Payment Method
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=0)
input_data['PaymentMethod_Credit card (automatic)'] = (payment_method == 'Credit card (automatic)')
input_data['PaymentMethod_Electronic check'] = (payment_method == 'Electronic check')
input_data['PaymentMethod_Mailed check'] = (payment_method == 'Mailed check')

# Convert input data to a DataFrame, ensuring correct order and types
# Initialize all selected features to their default (False for bool, 0 for int/float) first
processed_input = pd.DataFrame(0, index=[0], columns=selected_features)

for feature in selected_features:
    if feature in input_data:
        processed_input[feature] = input_data[feature]

# Ensure boolean features are correctly cast as bool if not already
for col in processed_input.columns:
    if processed_input[col].dtype == 'bool': # Only convert if it's already boolean
        processed_input[col] = processed_input[col].astype(bool)


if st.button('Predict Churn'):
    try:
        prediction = log_reg_model.predict(processed_input[selected_features]) # Ensure order of columns
        prediction_proba = log_reg_model.predict_proba(processed_input[selected_features])

        churn_status = 'Yes' if prediction[0] == 1 else 'No'
        churn_probability = prediction_proba[0][1] * 100 # Probability of churning (class 1)

        st.success(f"Prediction: Customer will {'**CHURN**' if churn_status == 'Yes' else '**NOT CHURN**'}")
        st.info(f"Probability of Churn: {churn_probability:.2f}%")

        if churn_status == 'Yes':
            st.warning("This customer is predicted to churn. Consider retention strategies.")
        else:
            st.success("This customer is predicted to remain.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and the loaded model.")
