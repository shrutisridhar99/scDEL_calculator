import streamlit as st
import pandas as pd
import numpy as np
import os 

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
    


def load_all_models(directory):
    models = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            model_name = filename.replace('_Model.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(directory, filename)
            models[model_name] = load_model(model_path)
    return models    

st.set_page_config(page_title="scDEL Calculator", page_icon="🧬", layout="wide")

st.title("scDEL Calculator")

models = load_all_models('model_test')


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
        
        # Function to display model results and get predictions
        def display_model_results(models, columns):
            predictions = {}
            for i, (model_name, model) in enumerate(models.items()):
                column = columns[i % len(columns)]
                probability = model.predict_proba(input_data)[0][1]
                predictions[model_name] = probability
                column.markdown(
                    f"<div style='background-color: #080d07; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<span style='color: #1e88e5; font-weight: bold;'>{model_name}:</span> "
                    f"<span style='color: #43a047; font-weight: bold;'>{probability:.2%}</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
            return predictions
        
        predictions = display_model_results(models, [col1, col2])
        
        # Determine majority prediction
        majority_high = sum(prob > 0.5 for prob in predictions.values()) > len(predictions) / 2
        majority_predictions = {name: prob for name, prob in predictions.items() if (prob > 0.5) == majority_high}
        minority_predictions = {name: prob for name, prob in predictions.items() if (prob > 0.5) != majority_high}
        
        # Calculate average of majority predictions
        majority_avg = sum(majority_predictions.values()) / len(majority_predictions)
        
        # Determine risk level and background color
        if majority_avg < 0.33:
            risk_level = "Low"
            bg_color = "#4caf50"  # Green
        elif majority_avg < 0.66:
            risk_level = "Medium"
            bg_color = "#ffd54f"  # Yellow
        else:
            risk_level = "High"
            bg_color = "#ef5350"  # Red
        
        # Display final prediction
        st.markdown("---")
        st.subheader("Final Prediction:")
        st.markdown(f"<div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>"
                    f"<span style='color: #ffffff; font-size: 24px; font-weight: bold;'>"
                    f"{risk_level} Risk of Relapse</span><br>"
                    f"<span style='color: #ffffff; font-size: 18px;'>Average Probability: {majority_avg:.2%}</span>"
                    "</div>", unsafe_allow_html=True)
        
        # Display majority and minority models
        st.markdown("<span style='color: #4caf50; font-weight: bold;'>Models in Majority:</span>", unsafe_allow_html=True)
        for name in majority_predictions.keys():
            st.markdown(f"- {name}")
        
        st.markdown("<br><span style='color: #ff7043; font-weight: bold;'>Models in Minority:</span>", unsafe_allow_html=True)
        for name in minority_predictions.keys():
            st.markdown(f"- {name}")
        
    except Exception as e:
        st.error(f"An error occurred while calculating the relapse risk: {str(e)}")
        st.error("Please ensure all inputs are valid and try again.")



        
st.sidebar.title("About")
st.sidebar.info(
    "This app calculates the relapse risk for DLBCL patients based on the scDEL model. "
    "Enter the required information and click 'Calculate Relapse Risk' to get the result."
)