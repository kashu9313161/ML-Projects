def create_features(df):

    df = df.copy()

    # returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # moving averages
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()

    df["price_to_sma20"] = (
        df["Close"] / df["sma_20"]
    ) - 1

    # volume
    df["volume_change"] = df["Volume"].pct_change()

    # range
    df["daily_range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    # target
    df["target"] = (
        df["Close"].shift(-1)
        / df["Close"]
    ) - 1

    # remove rows with missing values
    df = df.dropna()

    return df