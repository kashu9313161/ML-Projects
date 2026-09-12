from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import predict_price


app = FastAPI(
    title="House Price Prediction API",
    description="Ames Housing price prediction using XGBoost",
    version="1.0.0"
)


# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class HouseInput(BaseModel):

    # Property / Land
    MS_SubClass: float
    MS_Zoning: str
    Lot_Frontage: float | None = None
    Lot_Area: float
    Street: str
    Alley: str | None = None
    Lot_Shape: str
    Land_Contour: str
    Utilities: str
    Lot_Config: str
    Land_Slope: str

    # Location
    Neighborhood: str

    # Building
    Condition_1: str
    Condition_2: str
    Bldg_Type: str
    House_Style: str

    # Quality
    Overall_Qual: int
    Overall_Cond: int

    # Age
    Year_Built: int
    Year_Remod_Add: int

    # Exterior
    Roof_Style: str
    Roof_Matl: str
    Exterior_1st: str
    Exterior_2nd: str
    Mas_Vnr_Type: str | None = None
    Mas_Vnr_Area: float | None = None
    Exter_Qual: str
    Exter_Cond: str
    Foundation: str

    # Basement
    Bsmt_Qual: str | None = None
    Bsmt_Cond: str | None = None
    Bsmt_Exposure: str | None = None
    BsmtFin_Type_1: str | None = None
    BsmtFin_SF_1: float | None = None
    BsmtFin_Type_2: str | None = None
    BsmtFin_SF_2: float | None = None
    Bsmt_Unf_SF: float | None = None
    Total_Bsmt_SF: float | None = None

    # Utilities / Heating
    Heating: str
    Heating_QC: str
    Central_Air: str
    Electrical: str

    # Living Space
    First_Flr_SF: float
    Second_Flr_SF: float
    Low_Qual_Fin_SF: float
    Gr_Liv_Area: float

    # Rooms
    Bsmt_Full_Bath: float | None = None
    Bsmt_Half_Bath: float | None = None
    Full_Bath: float
    Half_Bath: float
    Bedroom_AbvGr: int
    Kitchen_AbvGr: int
    Kitchen_Qual: str
    TotRms_AbvGrd: int

    # Fireplace
    Functional: str
    Fireplaces: int
    Fireplace_Qu: str | None = None

    # Garage
    Garage_Type: str | None = None
    Garage_Yr_Blt: float | None = None
    Garage_Finish: str | None = None
    Garage_Cars: float | None = None
    Garage_Area: float | None = None
    Garage_Qual: str | None = None
    Garage_Cond: str | None = None

    # Outdoor
    Paved_Drive: str
    Wood_Deck_SF: float
    Open_Porch_SF: float
    Enclosed_Porch: float
    ThreeSsn_Porch: float
    Screen_Porch: float

    # Amenities
    Pool_Area: float
    Pool_QC: str | None = None
    Fence: str | None = None
    Misc_Feature: str | None = None
    Misc_Val: float

    # Sale
    Mo_Sold: int
    Yr_Sold: int
    Sale_Type: str
    Sale_Condition: str


# --------------------------------------------------
# Convert API names → Dataset names
# --------------------------------------------------

COLUMN_MAP = {
    # Property / Land
    "MS_SubClass": "MS SubClass",
    "MS_Zoning": "MS Zoning",
    "Lot_Frontage": "Lot Frontage",
    "Lot_Area": "Lot Area",
    "Street": "Street",
    "Alley": "Alley",
    "Lot_Shape": "Lot Shape",
    "Land_Contour": "Land Contour",
    "Utilities": "Utilities",
    "Lot_Config": "Lot Config",
    "Land_Slope": "Land Slope",

    # Location
    "Neighborhood": "Neighborhood",

    # Building
    "Condition_1": "Condition 1",
    "Condition_2": "Condition 2",
    "Bldg_Type": "Bldg Type",
    "House_Style": "House Style",

    # Quality
    "Overall_Qual": "Overall Qual",
    "Overall_Cond": "Overall Cond",

    # Age
    "Year_Built": "Year Built",
    "Year_Remod_Add": "Year Remod/Add",

    # Exterior
    "Roof_Style": "Roof Style",
    "Roof_Matl": "Roof Matl",
    "Exterior_1st": "Exterior 1st",
    "Exterior_2nd": "Exterior 2nd",
    "Mas_Vnr_Type": "Mas Vnr Type",
    "Mas_Vnr_Area": "Mas Vnr Area",
    "Exter_Qual": "Exter Qual",
    "Exter_Cond": "Exter Cond",
    "Foundation": "Foundation",

    # Basement
    "Bsmt_Qual": "Bsmt Qual",
    "Bsmt_Cond": "Bsmt Cond",
    "Bsmt_Exposure": "Bsmt Exposure",
    "BsmtFin_Type_1": "BsmtFin Type 1",
    "BsmtFin_SF_1": "BsmtFin SF 1",
    "BsmtFin_Type_2": "BsmtFin Type 2",
    "BsmtFin_SF_2": "BsmtFin SF 2",
    "Bsmt_Unf_SF": "Bsmt Unf SF",
    "Total_Bsmt_SF": "Total Bsmt SF",

    # Heating / Utilities
    "Heating": "Heating",
    "Heating_QC": "Heating QC",
    "Central_Air": "Central Air",
    "Electrical": "Electrical",

    # Living Space
    "First_Flr_SF": "1st Flr SF",
    "Second_Flr_SF": "2nd Flr SF",
    "Low_Qual_Fin_SF": "Low Qual Fin SF",
    "Gr_Liv_Area": "Gr Liv Area",

    # Rooms
    "Bsmt_Full_Bath": "Bsmt Full Bath",
    "Bsmt_Half_Bath": "Bsmt Half Bath",
    "Full_Bath": "Full Bath",
    "Half_Bath": "Half Bath",
    "Bedroom_AbvGr": "Bedroom AbvGr",
    "Kitchen_AbvGr": "Kitchen AbvGr",
    "Kitchen_Qual": "Kitchen Qual",
    "TotRms_AbvGrd": "TotRms AbvGrd",

    # Functional / Fireplace
    "Functional": "Functional",
    "Fireplaces": "Fireplaces",
    "Fireplace_Qu": "Fireplace Qu",

    # Garage
    "Garage_Type": "Garage Type",
    "Garage_Yr_Blt": "Garage Yr Blt",
    "Garage_Finish": "Garage Finish",
    "Garage_Cars": "Garage Cars",
    "Garage_Area": "Garage Area",
    "Garage_Qual": "Garage Qual",
    "Garage_Cond": "Garage Cond",

    # Outdoor
    "Paved_Drive": "Paved Drive",
    "Wood_Deck_SF": "Wood Deck SF",
    "Open_Porch_SF": "Open Porch SF",
    "Enclosed_Porch": "Enclosed Porch",
    "ThreeSsn_Porch": "3Ssn Porch",
    "Screen_Porch": "Screen Porch",

    # Amenities
    "Pool_Area": "Pool Area",
    "Pool_QC": "Pool QC",
    "Fence": "Fence",
    "Misc_Feature": "Misc Feature",
    "Misc_Val": "Misc Val",

    # Sale
    "Mo_Sold": "Mo Sold",
    "Yr_Sold": "Yr Sold",
    "Sale_Type": "Sale Type",
    "Sale_Condition": "Sale Condition",
}


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "XGBoost"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(house: HouseInput):

    # Pydantic → dictionary
    data = house.model_dump()

    # API names → original dataset names
    data = {
        COLUMN_MAP.get(key, key): value
        for key, value in data.items()
    }

    # Dictionary → DataFrame
    house_df = pd.DataFrame([data])

    # # Validation input columns
    # print("\nAPI input columns:")
    # print(house_df.columns.tolist())

    # Prediction
    predicted_price = predict_price(house_df)

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "USD"
    }