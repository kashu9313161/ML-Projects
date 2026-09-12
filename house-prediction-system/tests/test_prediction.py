import pandas as pd

from src.predict import predict_price


def test_prediction():

    df = pd.read_csv(
        "data/raw/AmesHousing.csv"
    )

    sample_house = df.drop(
        columns=["SalePrice"]
    ).iloc[[0]]

    prediction = predict_price(sample_house)

    # Prediction should be a number
    assert isinstance(prediction, float)

    # House price should be positive
    assert prediction > 0