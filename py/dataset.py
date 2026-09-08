from data_loader import download_stock_data
from features import create_features

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "price_to_sma20",
    "volume_change",
    "daily_range"
]


def create_dataset(ticker):

    df = download_stock_data(ticker)

    df = create_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    return X, y, df


def time_series_split(X, y, test_size=0.2):

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X, y, df = create_dataset("AAPL")

    X_train, X_test, y_train, y_test = time_series_split(X, y)

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nTraining period:")
    print(df.loc[X_train.index, "Date"].min())
    print(df.loc[X_train.index, "Date"].max())

    print("\nTesting period:")
    print(df.loc[X_test.index, "Date"].min())
    print(df.loc[X_test.index, "Date"].max())