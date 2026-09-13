import pandas as pd

from src.predict import predict_price


def test_prediction():

    sample_house = pd.DataFrame([{
        "Overall Qual": 7,
        "Overall Cond": 5,
        "Gr Liv Area": 1800,
        "1st Flr SF": 1000,
        "2nd Flr SF": 800,
        "Total Bsmt SF": 1000,
        "Garage Cars": 2,
        "Garage Area": 500,
        "Full Bath": 2,
        "Half Bath": 1,
        "Bedroom AbvGr": 3,
        "TotRms AbvGrd": 7,
        "Year Built": 2005,
        "Year Remod/Add": 2005,
        "Neighborhood": "NAmes",
        "Kitchen Qual": "TA",
        "Exter Qual": "TA",
        "Lot Area": 10000,
        "Fireplaces": 1
    }])

    prediction = predict_price(sample_house)

    assert isinstance(prediction, float)
    assert prediction > 0