import pandas as pd

from src.feature_engineering import create_features


def test_feature_engineering():

    df = pd.read_csv("data/raw/AmesHousing.csv")

    result = create_features(df)

    # Original identification columns should be removed
    assert "Order" not in result.columns
    assert "PID" not in result.columns

    # Engineered features should exist
    expected_features = [
        "TotalSF",
        "TotalBathrooms",
        "TotalPorchSF",
        "TotalBsmtFinSF",
        "HouseAge",
        "YearsSinceRemodel",
        "GarageAge",
        "Log_Lot_Area",
        "Log_TotalSF",
        "Log_Gr_Liv_Area",
    ]

    for feature in expected_features:
        assert feature in result.columns

    # Age features shouldn't be negative
    assert (result["HouseAge"] >= 0).all()
    assert (result["YearsSinceRemodel"] >= 0).all()
    # GarageAge can be NaN when Garage Yr Blt is missing,
    # but existing values shouldn't be negative.
    assert (result["GarageAge"].dropna() >= 0).all()