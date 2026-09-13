from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MARKET INDICES
# ============================================================

MARKET_INDICES = {
    "^NSEI": "NIFTY50",
    "^NSEBANK": "BANKNIFTY",
}


# ============================================================
# DOWNLOAD INDEX DATA
# ============================================================

def download_index_data(
    symbol,
    name,
    period="5y"
):

    print(
        f"\nDownloading {name} ({symbol})..."
    )

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    if data.empty:

        raise ValueError(
            f"No data found for {symbol}"
        )

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[
        columns
    ].copy()

    data.dropna(
        inplace=True
    )

    # --------------------------------------------------------
    # Market returns
    # --------------------------------------------------------

    data["Return_1D"] = (
        data["Close"].pct_change()
    )

    data["Return_5D"] = (
        data["Close"]
        .pct_change(5)
    )

    data["Return_20D"] = (
        data["Close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # Market moving averages
    # --------------------------------------------------------

    data["SMA_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["SMA_50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # Market volatility
    # --------------------------------------------------------

    data["Volatility_20D"] = (
        data["Return_1D"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # Position relative to moving averages
    # --------------------------------------------------------

    data["Price_vs_SMA20"] = (
        data["Close"]
        / data["SMA_20"]
        - 1
    )

    data["Price_vs_SMA50"] = (
        data["Close"]
        / data["SMA_50"]
        - 1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{name.lower()}.csv"
    )

    filepath = (
        DATA_DIR / filename
    )

    data.to_csv(
        filepath
    )

    return data, filepath


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("MARKET CONTEXT DATA")
    print("=" * 70)

    successful = 0
    failed = 0

    for symbol, name in MARKET_INDICES.items():

        try:

            data, filepath = (
                download_index_data(
                    symbol,
                    name
                )
            )

            latest_price = (
                data["Close"].iloc[-1]
            )

            latest_return = (
                data["Return_1D"].iloc[-1]
                * 100
            )

            print(
                f"{name} | "
                f"Price: {latest_price:.2f} | "
                f"1D: {latest_return:+.2f}% | "
                f"Rows: {len(data)}"
            )

            print(
                f"Saved: {filepath}"
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {name}: {e}"
            )

            failed += 1

    print("\n")
    print("=" * 70)
    print("MARKET CONTEXT DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


if __name__ == "__main__":
    main()