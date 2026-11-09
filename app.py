
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
MODEL_PATH = Path(__file__).with_name("churn_model.joblib")

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("churn_model.joblib was not found. Keep it in the same folder as app.py.")
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle['pipeline']
feature_columns = bundle['feature_columns']

st.title("Customer Churn Predictor")
st.write("Estimate churn risk and help the retention team prioritise customer follow-up.")
st.caption("This model provides an estimate, not a guarantee. Use it alongside customer context and staff judgement.")

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider(
        "Churn alert threshold", min_value=0.10, max_value=0.90,
        value=float(bundle.get('prediction_threshold', 0.50)), step=0.05,
        help="Lower thresholds usually catch more potential churners but create more false alerts."
    )

st.subheader("Customer details")
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    with col2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=50000.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    customer = {
        'gender': gender,
        'SeniorCitizen': int(senior),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': float(monthly),
        'TotalCharges': float(total)
    }
    input_df = pd.DataFrame([customer], columns=feature_columns)
    probability = float(model.predict_proba(input_df)[0, 1])
    predicted_churn = probability >= threshold

    if probability >= 0.60:
        risk = "High"
    elif probability >= 0.30:
        risk = "Medium"
    else:
        risk = "Low"

    st.divider()
    st.subheader("Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Estimated churn probability", f"{probability:.1%}")
    metric_col2.metric("Risk band", risk)
    metric_col3.metric("Alert at selected threshold", "Yes" if predicted_churn else "No")
    st.progress(min(max(probability, 0.0), 1.0))

    actions = []
    if contract == "Month-to-month":
        actions.append("Discuss a suitable annual contract or loyalty incentive; explain the savings and terms clearly.")
    if tenure < 12:
        actions.append("Prioritise onboarding: arrange a check-in, confirm service quality, and resolve setup issues early.")
    if payment == "Electronic check":
        actions.append("Offer help switching to an automatic payment option if the customer prefers it.")
    if monthly >= 80:
        actions.append("Review the bill and plan fit; explain charges and offer a lower-cost package if appropriate.")
    if tech_support == "No" and internet_service != "No":
        actions.append("Explain available technical support options and check whether unresolved issues are driving dissatisfaction.")
    if online_security == "No" and internet_service != "No":
        actions.append("Check whether the customer understands the available online security options and whether they are relevant.")
    if not actions:
        actions.append("Use a personalised service check-in and ask about any unresolved issues before offering a discount.")

    st.subheader("Suggested retention actions")
    for action in actions[:4]:
        st.write(f"- {action}")
    st.info("Use this estimate to prioritise a conversation. Do not automatically change a customer's plan or price based only on this score.")

st.divider()
st.caption(f"Model: {bundle.get('model_name', 'Trained classifier')} | Prediction threshold: {threshold:.2f}")
