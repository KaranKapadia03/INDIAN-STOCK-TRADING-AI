from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# RELATIVE STRENGTH FEATURES
# ============================================================

def add_relative_strength(data):

    data = data.copy()

    # --------------------------------------------------------
    # Stock vs NIFTY
    # --------------------------------------------------------

    data["Relative_5D_vs_NIFTY"] = (
        data["Return_5D"]
        - data["NIFTY_Return_5D"]
    )

    data["Relative_20D_vs_NIFTY"] = (
        data["Return_20D"]
        - data["NIFTY_Return_20D"]
    )

    # --------------------------------------------------------
    # Stock vs BANK NIFTY
    # --------------------------------------------------------

    data["Relative_5D_vs_BANKNIFTY"] = (
        data["Return_5D"]
        - data["BANKNIFTY_Return_5D"]
    )

    data["Relative_20D_vs_BANKNIFTY"] = (
        data["Return_20D"]
        - data["BANKNIFTY_Return_20D"]
    )

    # --------------------------------------------------------
    # Relative volatility
    #
    # How volatile is the stock compared with the market?
    # --------------------------------------------------------

    data["Relative_Volatility_NIFTY"] = (
        data["Volatility_20D"]
        / data["NIFTY_Volatility_20D"]
    )

    data["Relative_Volatility_BANKNIFTY"] = (
        data["Volatility_20D"]
        / data["BANKNIFTY_Volatility_20D"]
    )

    # --------------------------------------------------------
    # Relative price strength
    # --------------------------------------------------------

    data["Relative_Price_vs_NIFTY"] = (
        data["Price_vs_SMA20"]
        - data["NIFTY_Price_vs_SMA20"]
    )

    data["Relative_Price_vs_BANKNIFTY"] = (
        data["Price_vs_SMA20"]
        - data["BANKNIFTY_Price_vs_SMA20"]
    )

    # --------------------------------------------------------
    # Market trend alignment
    #
    # Positive = stock and market have stronger alignment
    # Negative = stock is weaker than the market
    # --------------------------------------------------------

    data["Trend_Strength_vs_NIFTY"] = (
        data["SMA20_vs_SMA50"]
        - (
            data["NIFTY_SMA_20"]
            / data["NIFTY_SMA_50"]
            - 1
        )
    )

    data["Trend_Strength_vs_BANKNIFTY"] = (
        data["SMA20_vs_SMA50"]
        - (
            data["BANKNIFTY_SMA_20"]
            / data["BANKNIFTY_SMA_50"]
            - 1
        )
    )

    return data


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(filepath):

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data = add_relative_strength(
        data
    )

    data.dropna(
        inplace=True
    )

    data.to_csv(
        filepath
    )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("ADDING RELATIVE STRENGTH FEATURES")
    print("=" * 80)

    stock_files = sorted(
        FEATURES_DIR.glob(
            "*_features.csv"
        )
    )

    successful = 0
    failed = 0

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
                filepath
            )

            print(
                f"Rows: {len(data)}"
            )

            print(
                f"Columns: {len(data.columns)}"
            )

            print(
                "Relative strength features added."
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

            failed += 1

    print("\n")
    print("=" * 80)
    print("RELATIVE STRENGTH INTEGRATION COMPLETE")
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