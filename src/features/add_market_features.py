from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# MARKET FEATURE COLUMNS
# ============================================================

MARKET_COLUMNS = [
    "Close",
    "Return_1D",
    "Return_5D",
    "Return_20D",
    "SMA_20",
    "SMA_50",
    "Volatility_20D",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
]


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    nifty = pd.read_csv(
        RAW_DIR / "nifty50.csv",
        index_col=0,
        parse_dates=True
    )

    banknifty = pd.read_csv(
        RAW_DIR / "banknifty.csv",
        index_col=0,
        parse_dates=True
    )

    return nifty, banknifty


# ============================================================
# ADD MARKET FEATURES
# ============================================================

def add_market_features(
    stock_data,
    nifty,
    banknifty
):

    stock_data = stock_data.copy()

    # --------------------------------------------------------
    # Keep only the required market columns
    # --------------------------------------------------------

    nifty_features = nifty[
        MARKET_COLUMNS
    ].copy()

    banknifty_features = banknifty[
        MARKET_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    nifty_features.columns = [
        f"NIFTY_{column}"
        for column in MARKET_COLUMNS
    ]

    banknifty_features.columns = [
        f"BANKNIFTY_{column}"
        for column in MARKET_COLUMNS
    ]

    # --------------------------------------------------------
    # Align by trading date
    #
    # This is important because the stock and index may have
    # slightly different trading-day counts.
    # --------------------------------------------------------

    stock_data = stock_data.join(
        nifty_features,
        how="left"
    )

    stock_data = stock_data.join(
        banknifty_features,
        how="left"
    )

    return stock_data


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(filepath, nifty, banknifty):

    stock_data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    stock_data = add_market_features(
        stock_data,
        nifty,
        banknifty
    )

    # --------------------------------------------------------
    # Remove rows where market data isn't available
    # --------------------------------------------------------

    market_columns = (
        [
            f"NIFTY_{column}"
            for column in MARKET_COLUMNS
        ]
        +
        [
            f"BANKNIFTY_{column}"
            for column in MARKET_COLUMNS
        ]
    )

    stock_data.dropna(
        subset=market_columns,
        inplace=True
    )

    stock_data.to_csv(
        filepath
    )

    return stock_data


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("ADDING MARKET CONTEXT FEATURES")
    print("=" * 80)

    # --------------------------------------------------------
    # Load market data
    # --------------------------------------------------------

    nifty, banknifty = load_market_data()

    print(
        f"\nNIFTY rows: {len(nifty)}"
    )

    print(
        f"BANKNIFTY rows: {len(banknifty)}"
    )

    # --------------------------------------------------------
    # Find stock feature files
    # --------------------------------------------------------

    stock_files = sorted(
        FEATURES_DIR.glob(
            "*_features.csv"
        )
    )

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every stock
    # --------------------------------------------------------

    for filepath in stock_files:

        symbol = (
            filepath.stem
            .replace("_features", "")
            .replace("_", ".")
        )

        print(
            f"\nProcessing {symbol}..."
        )

        try:

            data = process_stock(
                filepath,
                nifty,
                banknifty
            )

            print(
                f"Rows: {len(data)}"
            )

            print(
                f"Columns: {len(data.columns)}"
            )

            print(
                "Market features added successfully."
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MARKET FEATURE INTEGRATION COMPLETE")
    print("=" * 80)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total: {len(stock_files)}"
    )

    print("\n")


if __name__ == "__main__":
    main()