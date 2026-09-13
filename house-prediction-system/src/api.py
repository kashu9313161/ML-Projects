from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np


# --------------------------------
# FastAPI application
# --------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description="Ames Housing price prediction using XGBoost",
    version="1.0.0"
)


# --------------------------------
# Load application model
# --------------------------------

MODEL_PATH = "models/house_price_app_xgb.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------
# Input schema
# --------------------------------

class HouseInput(BaseModel):

    overall_qual: int
    overall_cond: int

    gr_liv_area: float
    first_flr_sf: float
    second_flr_sf: float
    total_bsmt_sf: float

    garage_cars: float
    garage_area: float

    full_bath: float
    half_bath: float

    bedroom_abvgr: int
    tot_rms_abvgrd: int

    year_built: int
    year_remod_add: int

    neighborhood: str
    kitchen_qual: str
    exter_qual: str

    lot_area: float
    fireplaces: int


# --------------------------------
# Health check
# --------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "XGBoost Application Model"
    }


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(house: HouseInput):

    data = house.model_dump()

    # Convert API names → dataset names
    data = {
        "Overall Qual": data["overall_qual"],
        "Overall Cond": data["overall_cond"],

        "Gr Liv Area": data["gr_liv_area"],
        "1st Flr SF": data["first_flr_sf"],
        "2nd Flr SF": data["second_flr_sf"],
        "Total Bsmt SF": data["total_bsmt_sf"],

        "Garage Cars": data["garage_cars"],
        "Garage Area": data["garage_area"],

        "Full Bath": data["full_bath"],
        "Half Bath": data["half_bath"],

        "Bedroom AbvGr": data["bedroom_abvgr"],
        "TotRms AbvGrd": data["tot_rms_abvgrd"],

        "Year Built": data["year_built"],
        "Year Remod/Add": data["year_remod_add"],

        "Neighborhood": data["neighborhood"],
        "Kitchen Qual": data["kitchen_qual"],
        "Exter Qual": data["exter_qual"],

        "Lot Area": data["lot_area"],
        "Fireplaces": data["fireplaces"]
    }

    # Convert dictionary → DataFrame
    house_df = pd.DataFrame([data])

    # Predict log(price)
    prediction_log = model.predict(house_df)

    # Convert log(price) → actual price
    predicted_price = np.expm1(prediction_log[0])

    return {
        "predicted_price": round(float(predicted_price), 2),
        "currency": "USD"
    }