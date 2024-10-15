import streamlit as st
import numpy as np
import pandas as pd


@st.cache_resource
def load_model():
    with open("model/random_forest_model_2.pkl", "rb") as file:
        return pd.read_pickle(file)

model = load_model()

st.title("scDEL Calculator")

coo = st.selectbox("COO", [0, 1])

col1, col2 = st.columns(2)
with col1:
    myc_slider = st.slider("MYC Percentage", 0, 100, 50)
with col2:
    myc_text = st.text_input("MYC Percentage", value=str(myc_slider))
myc = int(myc_text) if myc_text.isdigit() else myc_slider


col1, col2 = st.columns(2)
with col1:
    bcl2_slider = st.slider("BCL2 Percentage", 0, 100, 50)
with col2:
    bcl2_text = st.text_input("BCL2 Percentage", value=str(bcl2_slider))
bcl2 = int(bcl2_text) if bcl2_text.isdigit() else bcl2_slider


col1, col2 = st.columns(2)
with col1:
    bcl6_slider = st.slider("BCL6 Percentage", 0, 100, 50)
with col2:
    bcl6_text = st.text_input("BCL6 Percentage", value=str(bcl6_slider))
bcl6 = int(bcl6_text) if bcl6_text.isdigit() else bcl6_slider

m2p6n = (myc * bcl2 * (100 - bcl6)) / (100 * 100)

st.write(f"Calculated MYC+ BCL2+ BCL6- %: {m2p6n:.2f}")

input_data = np.array([[m2p6n, coo, myc, bcl2, bcl6]])

if st.button("Calculate Relapse Risk"):
    probability = model.predict_proba(input_data)[0][1]  # Assuming binary classification
    st.write(f"Probability of Relapse Risk on R-CHOP at 2 years: {probability:.2%}")