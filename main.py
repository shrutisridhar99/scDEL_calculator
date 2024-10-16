import streamlit as st
import pandas as pd
import numpy as np


@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pd.read_pickle(file)

def validate_percentage(value):
    try:
        value = float(value)
        return max(0, min(100, value))
    except ValueError:
        return 50 

st.set_page_config(page_title="scDEL Calculator", page_icon="🧬", layout="wide")

st.title("scDEL Calculator")

model_GB = load_model('model_test/gradient_boosting_Model.pkl')
model_rf = load_model('model_test/random_forest_Model.pkl')
model_knn = load_model('model_test/k-nearest_neighbors_Model.pkl')
model_xgboost = load_model('model_test/xgboost_Model.pkl')

ov_model_GB = load_model('model_testov/gradient_boosting_Model.pkl')
ov_model_rf = load_model('model_testov/random_forest_Model.pkl')
ov_model_knn = load_model('model_testov/k-nearest_neighbors_Model.pkl')
ov_model_xgboost = load_model('model_testov/xgboost_Model.pkl')


coo = st.selectbox("Cell of Origin (Hans)", options=[0, 1], format_func=lambda x: "GCB" if x == 0 else "non-GCB")

def create_percentage_input(label, key):
    col1, col2 = st.columns([3, 1])
    with col1:
        slider_value = st.slider(f"{label} Percentage", 0, 100, 50, key=f"{key}_slider")
    with col2:
        text_value = st.number_input(f"{label} %", value=float(slider_value), min_value=0.0, max_value=100.0, step=0.1, key=f"{key}_text")
    return validate_percentage(text_value)

myc = create_percentage_input("MYC", "myc")
bcl2 = create_percentage_input("BCL2", "bcl2")
bcl6 = create_percentage_input("BCL6", "bcl6")

m2p6n = (myc * bcl2 * (100 - bcl6)) / (100 * 100)

st.write(f"Calculated MYC+ BCL2+ BCL6- %: {m2p6n:.2f}")

input_data = np.array([[m2p6n, coo, myc , bcl2, bcl6]])

if st.button("Calculate Relapse Risk"):
    try:
        st.subheader("Probability of Relapse Risk on R-CHOP at 2 years:")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Function to display model results
        def display_model_results(models, column, title):
            column.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
            for model_name, model in models:
                probability = model.predict_proba(input_data)[0][1]
                column.markdown(
                    f"<div style='background-color: #080d07; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<span style='color: #1e88e5; font-weight: bold;'>{model_name}:</span> "
                    f"<span style='color: #43a047; font-weight: bold;'>{probability:.2%}</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
        
        # Regular models
        regular_models = [
            ("Gradient Boosting", model_GB),
            ("Random Forest", model_rf),
            ("K-Nearest Neighbors", model_knn),
            ("XGBoost", model_xgboost)
        ]
        display_model_results(regular_models, col1, "Regular Models")
        
        # Oversampled models
        oversampled_models = [
            ("Gradient Boosting", ov_model_GB),
            ("Random Forest", ov_model_rf),
            ("K-Nearest Neighbors", ov_model_knn),
            ("XGBoost", ov_model_xgboost)
        ]
        display_model_results(oversampled_models, col2, "Oversampled Models")

    except Exception as e:
        st.error(f"An error occurred while calculating the relapse risk: {str(e)}")
        st.error("Please ensure all inputs are valid and try again.")


        
st.sidebar.title("About")
st.sidebar.info(
    "This app calculates the relapse risk for DLBCL patients based on the scDEL model. "
    "Enter the required information and click 'Calculate Relapse Risk' to get the result."
)