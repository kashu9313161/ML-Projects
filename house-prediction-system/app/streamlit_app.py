import streamlit as st
import requests


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("🏠 House Price Predictor")

st.write(
    "Enter the characteristics of a house "
    "to estimate its sale price."
)


# ---------------------------------------
# API URL
# ---------------------------------------

API_URL = "http://127.0.0.1:8000/predict"


# ---------------------------------------
# House Information
# ---------------------------------------

st.header("House Information")

col1, col2, col3 = st.columns(3)

with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )


with col2:

    overall_cond = st.slider(
        "Overall Condition",
        min_value=1,
        max_value=10,
        value=5
    )

    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=6,
        value=2
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )


with col3:

    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=500,
        max_value=250000,
        value=10000
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=6,
        value=2
    )

    bedroom_abvgr = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3
    )


# ---------------------------------------
# Prediction Button
# ---------------------------------------

st.divider()

if st.button(
    "Predict House Price",
    type="primary"
):

    st.info(
        "Connecting to the prediction API..."
    )

    # We will complete the API input mapping
    # after verifying this basic frontend.