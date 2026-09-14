import pandas as pd

from src.feature_engineering import create_features


def test_feature_engineering():

    df = pd.DataFrame({
        "Order": [1],
        "PID": [10001],
        "SalePrice": [200000],

        "Lot Area": [8000],
        "Gr Liv Area": [1700],

        "Total Bsmt SF": [1000],
        "1st Flr SF": [1200],
        "2nd Flr SF": [500],

        "Full Bath": [2],
        "Half Bath": [1],
        "Bsmt Full Bath": [1],
        "Bsmt Half Bath": [0],

        "Wood Deck SF": [100],
        "Open Porch SF": [50],
        "Enclosed Porch": [20],
        "3Ssn Porch": [0],
        "Screen Porch": [10],

        "BsmtFin SF 1": [500],
        "BsmtFin SF 2": [100],

        "Year Built": [2000],
        "Year Remod/Add": [2010],
        "Yr Sold": [2020],
        "Garage Yr Blt": [2000],
    })

    result = create_features(df)

    assert "TotalSF" in result.columns
    assert "TotalBathrooms" in result.columns
    assert "TotalPorchSF" in result.columns
    assert "TotalBsmtFinSF" in result.columns
    assert "HouseAge" in result.columns
    assert "YearsSinceRemodel" in result.columns
    assert "GarageAge" in result.columns

    assert result["TotalSF"].iloc[0] == 2700
    assert result["TotalBathrooms"].iloc[0] == 3.5
    assert result["TotalPorchSF"].iloc[0] == 180
    assert result["TotalBsmtFinSF"].iloc[0] == 600