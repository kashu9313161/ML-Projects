# Input house information
#         ↓
# Feature engineering
#         ↓
# Saved preprocessing
#         ↓
# Saved XGBoost
#         ↓
# Predicted SalePrice


    #             USER
    #              │
    #              ▼
    #       ┌─────────────┐
    #       │  Frontend   │
    #       └──────┬──────┘
    #              │
    #              │ JSON
    #              ▼
    #       ┌─────────────┐
    #       │   FastAPI   │
    #       │  /predict   │
    #       └──────┬──────┘
    #              │
    #              ▼
    #    house_price_xgb.joblib
    #              │
    #              ▼
    #       Predicted Price
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import create_features

MODEL_PATH = "models/house_price_xgb.joblib"

def load_model():
    """Load the trained ML pipeline."""
    return joblib.load(MODEL_PATH)

# Load model once when this module is imported
model = load_model()

def predict_price(house_data: pd.DataFrame) -> float:
    """
    Predict house price for new house data.
    """

    # Apply the same feature engineering used during training
    house_data = create_features(house_data)

    # Make prediction in log scale
    prediction_log = model.predict(house_data)

    # Convert log prediction back to original dollar scale
    prediction = np.expm1(prediction_log)

    return float(prediction[0])

# if __name__ == "__main__":
#     # Load one existing house as a test example
#     df = pd.read_csv("data/raw/AmesHousing.csv")

#     # Take one house
#     sample_house = df.drop(columns=["SalePrice"]).iloc[[0]]

#     predicted_price = predict_price(sample_house)

#     print("\n===== HOUSE PRICE PREDICTION ======")
#     print(f"Predicted price: ${predicted_price:,.2f}")