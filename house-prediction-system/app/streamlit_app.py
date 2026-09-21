import streamlit as st
import requests
import os


# =================================
# Page configuration
# =================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# =================================
# API configuration
# =================================

# Local Docker Compose:
#     http://backend:8000
#
# Render:
#     API_URL environment variable should contain:
#     https://house-price-api-oghr.onrender.com

API_URL = os.getenv(
    "API_URL",
    "http://backend:8000"
).rstrip("/")

# Make sure /predict is present exactly once
if not API_URL.endswith("/predict"):
    API_URL += "/predict"


# =================================
# Header
# =================================

st.title("🏠 House Price Predictor")

st.write(
    "Enter the characteristics of a house to estimate its sale price."
)

st.divider()


# =================================
# Property Quality
# =================================

st.header("🏡 Property Quality")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall material and finish quality of the house."
    )

with col2:
    overall_cond = st.slider(
        "Overall Condition",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall condition of the house."
    )


# =================================
# Property Size
# =================================

st.header("📐 Property Size")

col1, col2, col3 = st.columns(3)

with col1:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100.0,
        max_value=10000.0,
        value=1500.0,
        step=50.0
    )

with col2:
    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=50.0
    )

with col3:
    second_flr_sf = st.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=50.0
    )


col1, col2 = st.columns(2)

with col1:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0.0,
        max_value=10000.0,
        value=800.0,
        step=50.0
    )

with col2:
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=500.0,
        max_value=250000.0,
        value=10000.0,
        step=500.0
    )


# =================================
# Rooms & Bathrooms
# =================================

st.header("🛏️ Rooms & Bathrooms")

col1, col2, col3, col4 = st.columns(4)

with col1:
    bedroom_abvgr = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

with col2:
    tot_rms_abvgrd = st.number_input(
        "Total Rooms",
        min_value=1,
        max_value=20,
        value=7,
        step=1
    )

with col3:
    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=1.0
    )

with col4:
    half_bath = st.number_input(
        "Half Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=1.0
    )


# =================================
# Garage
# =================================

st.header("🚗 Garage")

col1, col2 = st.columns(2)

with col1:
    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=1.0
    )

with col2:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0.0,
        max_value=3000.0,
        value=500.0,
        step=25.0
    )


# =================================
# House History
# =================================

st.header("🔨 House History")

col1, col2 = st.columns(2)

with col1:
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )

with col2:
    year_remod_add = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1
    )


# Validate construction/remodeling years
if year_remod_add < year_built:
    st.warning(
        "The remodeling year cannot be earlier than the construction year."
    )


# =================================
# Features
# =================================

st.header("✨ House Features")

col1, col2, col3 = st.columns(3)

with col1:
    fireplaces = st.number_input(
        "Fireplaces",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

with col2:

    kitchen_quality_options = {
        "Excellent": "Ex",
        "Good": "Gd",
        "Typical / Average": "TA",
        "Fair": "Fa",
        "Poor": "Po"
    }

    kitchen_qual_label = st.selectbox(
        "Kitchen Quality",
        list(kitchen_quality_options.keys())
    )

    kitchen_qual = kitchen_quality_options[kitchen_qual_label]


with col3:

    exter_quality_options = {
        "Excellent": "Ex",
        "Good": "Gd",
        "Typical / Average": "TA",
        "Fair": "Fa",
        "Poor": "Po"
    }

    exter_qual_label = st.selectbox(
        "Exterior Quality",
        list(exter_quality_options.keys())
    )

    exter_qual = exter_quality_options[exter_qual_label]


# =================================
# Location
# =================================

st.header("📍 Location")

neighborhood = st.selectbox(
    "Neighborhood",
    [
        "Blmngtn",
        "Blueste",
        "BrDale",
        "BrkSide",
        "ClearCr",
        "CollgCr",
        "Crawfor",
        "Edwards",
        "Gilbert",
        "Greens",
        "GrnHill",
        "IDOTRR",
        "Landmrk",
        "MeadowV",
        "Mitchel",
        "NAmes",
        "NoRidge",
        "NPkVill",
        "NridgHt",
        "NWAmes",
        "OldTown",
        "Sawyer",
        "SawyerW",
        "Somerst",
        "StoneBr",
        "SWISU",
        "Timber",
        "Veenker"
    ]
)


st.divider()


# =================================
# Prediction
# =================================

if st.button(
    "💰 Predict House Price",
    type="primary",
    use_container_width=True
):

    # ---------------------------------
    # Validation
    # ---------------------------------

    if year_remod_add < year_built:
        st.error(
            "Please enter a valid remodeling year."
        )
        st.stop()

    if first_flr_sf + second_flr_sf > gr_liv_area:
        st.error(
            "1st Floor Area + 2nd Floor Area cannot be "
            "greater than Total Living Area."
        )
        st.stop()

    if year_built > 2026:
        st.error(
            "Year Built cannot be in the future."
        )
        st.stop()


    # ---------------------------------
    # API payload
    # ---------------------------------

    payload = {

        "overall_qual": overall_qual,
        "overall_cond": overall_cond,

        "gr_liv_area": gr_liv_area,
        "first_flr_sf": first_flr_sf,
        "second_flr_sf": second_flr_sf,
        "total_bsmt_sf": total_bsmt_sf,
        "lot_area": lot_area,

        "garage_cars": garage_cars,
        "garage_area": garage_area,

        "full_bath": full_bath,
        "half_bath": half_bath,

        "bedroom_abvgr": bedroom_abvgr,
        "tot_rms_abvgrd": tot_rms_abvgrd,

        "year_built": year_built,
        "year_remod_add": year_remod_add,

        "neighborhood": neighborhood,

        "kitchen_qual": kitchen_qual,
        "exter_qual": exter_qual,

        "fireplaces": fireplaces
    }


    # ---------------------------------
    # Call FastAPI
    # ---------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )


        # ---------------------------------
        # Successful prediction
        # ---------------------------------

        if response.status_code == 200:

            result = response.json()

            predicted_price = result["predicted_price"]

            st.success(
                "Prediction generated successfully!"
            )

            st.markdown(
                f"""
                <div style="
                    padding: 30px;
                    border-radius: 15px;
                    border: 1px solid #ddd;
                    text-align: center;
                    margin-top: 20px;
                ">
                    <h2>🏠 Estimated House Price</h2>

                    <h1>${predicted_price:,.2f}</h1>

                    <p style="font-size: 16px;">
                        Prediction generated by our
                        XGBoost application model
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Model performance: R² = 0.9155 | "
                "MAE ≈ $15,085 | 19 input features"
            )


        # ---------------------------------
        # API returned an error
        # ---------------------------------

        else:

            st.error(
                f"API returned an error: {response.status_code}"
            )

            # Temporary debugging information
            st.write(
                "API URL being used:",
                API_URL
            )

            st.write(
                "API response:",
                response.text
            )


    # ---------------------------------
    # Connection error
    # ---------------------------------

    except requests.exceptions.ConnectionError as e:

        st.error(
            "Could not connect to the FastAPI server."
        )

        st.write(
            "API URL being used:",
            API_URL
        )

        st.write(
            "Connection error:",
            str(e)
        )


    # ---------------------------------
    # Other request errors
    # ---------------------------------

    except requests.exceptions.RequestException as e:

        st.error(
            "Request failed."
        )

        st.write(
            "API URL being used:",
            API_URL
        )

        st.write(
            "Error:",
            str(e)
        )