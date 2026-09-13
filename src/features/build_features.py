
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Make src/data available for importing
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_SRC_DIR = PROJECT_ROOT / "src" / "data"

sys.path.append(str(DATA_SRC_DIR))


from technical_indicators import add_technical_indicators
from stock_universe import get_stock_universe


# --------------------------------------------------
# Project directories
# --------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Build features for one stock
# --------------------------------------------------

def build_features(symbol: str) -> pd.DataFrame:
    """
    Load raw stock data and add technical indicators.
    """

    filename = symbol.replace(".", "_") + ".csv"

    raw_file = RAW_DATA_DIR / filename

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_file}"
        )

    data = pd.read_csv(
        raw_file,
        index_col=0,
        parse_dates=True
    )

    # Add technical indicators
    data = add_technical_indicators(data)

    # Remove rows where indicators
    # cannot yet be calculated
    data.dropna(inplace=True)

    return data


# --------------------------------------------------
# Save processed features
# --------------------------------------------------

def save_features(
    data: pd.DataFrame,
    symbol: str
) -> str:
    """
    Save processed stock feature data.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    output_file = (
        PROCESSED_DATA_DIR / filename
    )

    data.to_csv(output_file)

    return str(output_file)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    stocks = get_stock_universe()

    print("\n" + "=" * 60)

    print("BUILDING FEATURES FOR ALL STOCKS")

    print("=" * 60)

    successful = 0

    failed = 0

    # Process every stock
    for symbol in stocks["symbol"]:

        print(
            f"\nProcessing {symbol}..."
        )

        try:

            data = build_features(
                symbol
            )

            output_file = save_features(
                data,
                symbol
            )

            print(
                f"Rows: {len(data)}"
            )

            print(
                f"Columns: {len(data.columns)}"
            )

            print(
                f"Saved: {output_file}"
            )

            successful += 1

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

            failed += 1

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE BUILD COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total: {len(stocks)}"
    )


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()
