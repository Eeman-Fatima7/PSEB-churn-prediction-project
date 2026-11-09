# Customer Churn Predictor

Live app: https://your-app-name.streamlit.app

Predicts which telecom customers may leave and suggests human-reviewed retention actions.

## Business problem

Customer retention can cost less than acquiring replacement customers. This project explores the Telco Customer Churn dataset and builds a recall-focused classifier to help a retention team prioritise follow-up.

## Dataset and cleaning

- Source: Telco Customer Churn dataset (Kaggle)
- Rows after cleaning: 7,043
- Original columns: 21
- Churn rate: 26.5%
- `TotalCharges` was converted from text to numeric. The 11 blank values in this supplied dataset correspond to tenure 0 and were set to 0.
- `customerID` was excluded from the model because it is an identifier.

## Key findings

Churn rates by contract:
- Month-to-month: 42.7% churn
- One year: 11.3% churn
- Two year: 2.8% churn

Churn rates by payment method:
- Electronic check: 45.3% churn
- Mailed check: 19.1% churn
- Bank transfer (automatic): 16.7% churn
- Credit card (automatic): 15.2% churn

- Median tenure for churned customers: 10 months
- Median tenure for retained customers: 38 months
- Median monthly charges for churned customers: $79.65
- Median monthly charges for retained customers: $64.43

## Model

- Compared Logistic Regression, Decision Tree, and Random Forest using 5-fold stratified cross-validation.
- Used class weighting to address class imbalance.
- Selected and tuned the model using recall as the primary score.
- Final model: Logistic Regression

Held-out test metrics:
- Accuracy: 0.738
- Precision: 0.504
- Recall: 0.783
- F1: 0.614
- ROC-AUC: 0.842

Recall is prioritised because failing to identify a customer who churns can mean a missed retention opportunity. A lower alert threshold can catch more churners but may increase false positives.

## Business recommendations

1. Review month-to-month customers for suitable contract or loyalty offers.
2. Improve first-year onboarding and early customer success check-ins.
3. Investigate payment friction among electronic-check users and make alternatives easy to understand.
4. Review high-bill accounts for plan fit, unexpected charges, and value perception.

These are associations in historical data, not proof of causation. Test retention actions with controlled campaigns and monitor outcomes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py` and `churn_model.joblib` in the same directory.

## Deploy

1. Push the project files to a public GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the main file and deploy.
4. Replace the Live app URL at the top of this README with the deployed URL.

## Built with

Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit, and joblib. Dataset from Kaggle.
