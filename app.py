import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

st.title('Loan Prediction App')
st.write('Welcome to the Loan Prediction App. Please enter the details to get the predicted loan amount')

# Get the user input
gross_income = st.number_input('Annual GrossIncome (USD)', 0.00, help='Enter your Annual Gross Income')
taxable_income = st.number_input('Annual TaxableIncome (USD)', 0.00, help='Enter your Annual Taxable Income')
nontaxable_income = st.number_input('Annual NontaxableIncome (USD)', 0.00, help='Enter your Annual Nontaxable Income')
total_deduction = st.number_input('Annual TotalDeduction (USD)', 0.00, help='Enter your Annual Total Deduction')
withholding_tax = st.number_input('Annual WithholdingTax (USD)', 0.00, help='Enter your Annual Withholding Tax')
net_worth = st.number_input('NetWorth (USD)', 0.00, help='Enter your Annual Net Worth')
previous_loan_amount = st.number_input('PreviousLoanAmount (USD)', 0.00, help='Enter the Previous Loan Amount')
repayment_years = st.number_input('RepaymentYears (Years)', 0, help='Enter the Repayment Years')
loan_interest = st.number_input('LoanInterest (%)', 0.00, 100.00, help='Enter the Loan Interest Rate')

monthly_rate = gross_income / 12
net_pay = gross_income - withholding_tax - total_deduction


entered_data = np.array([[gross_income, monthly_rate, taxable_income, nontaxable_income, total_deduction, withholding_tax, net_pay, net_worth, previous_loan_amount, repayment_years, loan_interest]]) 

if(st.button('Predict')):
    if(gross_income<=0 or net_worth<=0):
        st.error('Please enter valid values. Gross Income and Net Worth should be greater than 0')
    else:
        # Load the trained model
        model = tf.keras.models.load_model("model.keras")

        # Load the saved scalers
        scaler_x = joblib.load("scaler_x.pkl")
        scaler_y = joblib.load("scaler_y.pkl")

        # Load the saved scalers
        scaler_x = joblib.load("scaler_x.pkl")
        scaler_y = joblib.load("scaler_y.pkl")

        # Scale the input data
        entered_data_scaled = scaler_x.transform(entered_data)

        # Get predictions from the model
        prediction_scaled = model.predict(entered_data_scaled)

        # Convert prediction back to original scale
        prediction_original = scaler_y.inverse_transform(prediction_scaled)

        # Display the predicted loan amount
        st.write('The predicted loan amount is:', prediction_original[0][0])
        print("Prediction:", prediction_original[0][0])
