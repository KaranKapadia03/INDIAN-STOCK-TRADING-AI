import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable for the ML model.

    Target:
        BUY  = next-day return > +1%
        HOLD = next-day return between -1% and +1%
        SELL = next-day return < -1%
    """

    df = data.copy()

    # Calculate next-day return
    df["Next_Day_Return"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    ) * 100

    # Create BUY / HOLD / SELL labels
    df["Target"] = "HOLD"

    df.loc[
        df["Next_Day_Return"] > 1,
        "Target"
    ] = "BUY"

    df.loc[
        df["Next_Day_Return"] < -1,
        "Target"
    ] = "SELL"

    # Last row has no next-day data
    df.dropna(
        subset=["Next_Day_Return"],
        inplace=True
    )

    return df


def main():

    symbol = "RELIANCE.NS"

    filename = symbol.replace(".", "_") + "_features.csv"

    input_file = PROCESSED_DATA_DIR / filename

    print(f"\nLoading: {input_file}")

    data = pd.read_csv(
        input_file,
        index_col=0,
        parse_dates=True
    )

    data = create_target(data)

    output_file = (
        PROCESSED_DATA_DIR
        / symbol.replace(".", "_")
        / "training_data.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(output_file)

    print("\nTarget creation completed.")
    print("=" * 50)

    print("\nTarget distribution:")

    print(
        data["Target"].value_counts()
    )

    print("\nLatest rows:")

    print(
        data[
            [
                "Close",
                "Next_Day_Return",
                "Target"
            ]
        ].tail()
    )

    print("\nSaved to:")

    print(output_file)


if __name__ == "__main__":
    main()