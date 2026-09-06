import yfinance as yf
import pandas as pd


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "JPM",
    "NFLX"
]


def download_stock_data(ticker, start="2018-01-01", end=None):
    """
    Download historical OHLCV data for a stock.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # Remove yfinance column index name
    df.columns.name = None

    # Keep only the raw market data we need
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df = df[required_columns]

    # Basic validation
    df = df.dropna()

    df = df.sort_values("Date")

    return df


if __name__ == "__main__":

    data = download_stock_data("AAPL")

    print(data.head())
    print("\nShape:", data.shape)
    print("\nColumns:")
    print(data.columns)