import os
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_universe import get_stock_universe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_stock_data(
    symbol: str,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Download historical stock market data.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker symbol.
        Example: RELIANCE.NS

    period : str
        Historical period.
        Example: 1y, 2y, 5y, 10y

    Returns
    -------
    pd.DataFrame
        OHLCV market data.
    """

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(
            f"No market data found for {symbol}"
        )

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[columns].copy()

    data.dropna(inplace=True)

    return data


def save_stock_data(
    data: pd.DataFrame,
    symbol: str
) -> str:
    """
    Save stock data as a CSV file.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_symbol = symbol.replace(
        ".",
        "_"
    )

    filename = f"{safe_symbol}.csv"

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    data.to_csv(filepath)

    return filepath


def main():

    stocks = get_stock_universe()

    print("\n" + "=" * 70)
    print("DOWNLOADING 5 YEARS OF INDIAN STOCK DATA")
    print("=" * 70)

    successful = 0
    failed = 0

    for symbol in stocks["symbol"]:

        try:

            print(
                f"\nDownloading {symbol}..."
            )

            data = get_stock_data(
                symbol,
                period="5y"
            )

            filepath = save_stock_data(
                data,
                symbol
            )

            latest_price = (
                data["Close"].iloc[-1]
            )

            previous_price = (
                data["Close"].iloc[-2]
            )

            change_percent = (
                (
                    latest_price
                    - previous_price
                )
                / previous_price
            ) * 100

            print(
                f"{symbol} | "
                f"Latest Price: "
                f"₹{latest_price:.2f} | "
                f"Change: "
                f"{change_percent:.2f}% | "
                f"Rows: "
                f"{len(data)}"
            )

            print(
                f"Saved: {filepath}"
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

            failed += 1

    print("\n" + "=" * 70)
    print("MARKET DATA DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total: {len(stocks)}"
    )


if __name__ == "__main__":
    main()