import streamlit as st
import numpy as np
import pandas as pd
import pickle


# ============================================
# LOAD TRAINED MODEL
# ============================================

weights = np.load("model_weights.npy")
bias = np.load("model_bias.npy")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ============================================
# SIGMOID FUNCTION
# ============================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))    

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_probability(X):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# ============================================
# STREAMLIT PAGE
# ============================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to churn."
)

st.divider()

# ============================================
# CUSTOMER INPUTS
# ============================================

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    # ============================================
# REMAINING CUSTOMER INPUTS
# ============================================

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.01
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=0.01
    )

    # ============================================
# CREATE CUSTOMER DATAFRAME
# ============================================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# ============================================
# ONE-HOT ENCODING
# ============================================

customer_encoded = pd.get_dummies(
    customer_data,
    drop_first=True
)

# ============================================
# MATCH TRAINING FEATURES
# ============================================

customer_encoded = customer_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# ============================================
# SCALE NUMERICAL FEATURES
# ============================================

scale_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

customer_encoded[scale_columns] = scaler.transform(
    customer_encoded[scale_columns]
)

# ============================================
# PREDICTION
# ============================================

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    # Convert DataFrame to NumPy array
    X_customer = customer_encoded.values

    # Get churn probability
    probability = predict_probability(X_customer)

    # Convert array value to a normal number
    probability = float(probability[0][0])

    # Apply 0.5 threshold
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )