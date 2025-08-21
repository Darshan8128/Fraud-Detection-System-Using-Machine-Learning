import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load Saved Model (joblib)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🔎 Fraud Detection Web App")

st.markdown("Upload a dataset for predictions or enter transaction details manually.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    if st.button("Predict Fraud for Uploaded Data"):
        preds = model.predict(df)
        df["Fraud_Prediction"] = ["Fraud" if p == 1 else "Not Fraud" for p in preds]
        st.write("### Predictions", df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", mime="text/csv")

# Manual Input
st.write("### Or Enter Transaction Details Manually")

# Example fields – adjust based on your dataset columns
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=0.01)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=0.01)
with col2:
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=0.01)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=0.01)
    transactionType = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

# Predict Button
if st.button("Predict Fraud Manually"):
    input_data = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type": transactionType
    }])
    
    prediction = model.predict(input_data)[0]
    result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    st.subheader(result)
