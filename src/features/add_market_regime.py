from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = Path("data") / "processed"

NIFTY_SYMBOL = "^NSEI"

OUTPUT_SUFFIX = "_features_regime.csv"


# ============================================================
# DATE KEY
# ============================================================

def add_date_key(df):

    df = df.copy()

    dates = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True
    )

    # Convert every date into an integer day number.
    # This completely avoids pandas datetime precision issues.
    df["Date_Key"] = (
        dates.dt.year * 10000
        + dates.dt.month * 100
        + dates.dt.day
    )

    return df


# ============================================================
# DOWNLOAD NIFTY
# ============================================================

def download_nifty(start_date, end_date):

    print("\nDownloading NIFTY 50...")

    nifty = yf.Ticker(
        NIFTY_SYMBOL
    ).history(
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    if nifty.empty:
        raise RuntimeError(
            "Could not download NIFTY 50 data."
        )

    nifty = nifty.reset_index()

    nifty["Date"] = pd.to_datetime(
        nifty["Date"],
        errors="coerce",
        utc=True
    )

    nifty["Date_Key"] = (
        nifty["Date"].dt.year * 10000
        + nifty["Date"].dt.month * 100
        + nifty["Date"].dt.day
    )

    nifty = nifty.dropna(
        subset=["Date_Key"]
    )

    nifty["Date_Key"] = (
        nifty["Date_Key"]
        .astype("int64")
    )

    nifty = nifty.sort_values(
        "Date_Key"
    )

    nifty = nifty.drop_duplicates(
        subset=["Date_Key"]
    )

    return nifty


# ============================================================
# BUILD MARKET FEATURES
# ============================================================

def build_market_features(nifty):

    df = nifty[
        [
            "Date_Key",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ].copy()

    df = df.rename(
        columns={
            "Open": "NIFTY_Open",
            "High": "NIFTY_High",
            "Low": "NIFTY_Low",
            "Close": "NIFTY_Close",
            "Volume": "NIFTY_Volume"
        }
    )

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["NIFTY_Return_1D"] = (
        df["NIFTY_Close"]
        .pct_change(1)
    )

    df["NIFTY_Return_5D"] = (
        df["NIFTY_Close"]
        .pct_change(5)
    )

    df["NIFTY_Return_20D"] = (
        df["NIFTY_Close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    df["NIFTY_SMA_20"] = (
        df["NIFTY_Close"]
        .rolling(20)
        .mean()
    )

    df["NIFTY_SMA_50"] = (
        df["NIFTY_Close"]
        .rolling(50)
        .mean()
    )

    df["NIFTY_SMA_200"] = (
        df["NIFTY_Close"]
        .rolling(200)
        .mean()
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["NIFTY_Price_vs_SMA20"] = (
        df["NIFTY_Close"]
        / df["NIFTY_SMA_20"]
        - 1
    )

    df["NIFTY_Price_vs_SMA50"] = (
        df["NIFTY_Close"]
        / df["NIFTY_SMA_50"]
        - 1
    )

    df["NIFTY_Price_vs_SMA200"] = (
        df["NIFTY_Close"]
        / df["NIFTY_SMA_200"]
        - 1
    )

    df["NIFTY_SMA20_vs_SMA50"] = (
        df["NIFTY_SMA_20"]
        / df["NIFTY_SMA_50"]
        - 1
    )

    df["NIFTY_SMA50_vs_SMA200"] = (
        df["NIFTY_SMA_50"]
        / df["NIFTY_SMA_200"]
        - 1
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    daily_returns = (
        df["NIFTY_Close"]
        .pct_change()
    )

    df["NIFTY_Volatility_5D"] = (
        daily_returns
        .rolling(5)
        .std()
        * np.sqrt(252)
    )

    df["NIFTY_Volatility_20D"] = (
        daily_returns
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    bullish = (
        (df["NIFTY_Close"] > df["NIFTY_SMA_50"])
        &
        (df["NIFTY_SMA_50"] > df["NIFTY_SMA_200"])
    )

    bearish = (
        (df["NIFTY_Close"] < df["NIFTY_SMA_50"])
        &
        (df["NIFTY_SMA_50"] < df["NIFTY_SMA_200"])
    )

    df["Market_Regime"] = np.select(
        [
            bullish,
            bearish
        ],
        [
            "BULLISH",
            "BEARISH"
        ],
        default="NEUTRAL"
    )

    df["Market_Regime_Score"] = (
        df["Market_Regime"]
        .map(
            {
                "BULLISH": 1,
                "NEUTRAL": 0,
                "BEARISH": -1
            }
        )
    )

    return df


# ============================================================
# RELATIVE STRENGTH
# ============================================================

def add_relative_strength(df):

    df = df.copy()

    required = [
        "Return_1D",
        "Return_5D",
        "Return_20D",
        "NIFTY_Return_1D",
        "NIFTY_Return_5D",
        "NIFTY_Return_20D"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns after merge: {missing}"
        )

    df["Relative_Strength_1D"] = (
        df["Return_1D"]
        - df["NIFTY_Return_1D"]
    )

    df["Relative_Strength_5D"] = (
        df["Return_5D"]
        - df["NIFTY_Return_5D"]
    )

    df["Relative_Strength_20D"] = (
        df["Return_20D"]
        - df["NIFTY_Return_20D"]
    )

    return df


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(file_path, market):

    print(
        f"\nProcessing: {file_path.name}"
    )

    df = pd.read_csv(
        file_path
    )

    if "Date" not in df.columns:

        print(
            "  SKIPPED: Date column missing"
        )

        return False

    # --------------------------------------------------------
    # CREATE INTEGER DATE KEY
    # --------------------------------------------------------

    df = add_date_key(df)

    df = df.dropna(
        subset=["Date_Key"]
    )

    df["Date_Key"] = (
        df["Date_Key"]
        .astype("int64")
    )

    df = df.sort_values(
        "Date_Key"
    )

    df = df.drop_duplicates(
        subset=["Date_Key"]
    )

    # --------------------------------------------------------
    # REMOVE OLD MARKET FEATURES
    # --------------------------------------------------------

    columns_to_remove = [
        column
        for column in df.columns
        if (
            column.startswith("NIFTY_")
            or column.startswith("Relative_Strength_")
            or column == "Market_Regime"
            or column == "Market_Regime_Score"
        )
    ]

    if columns_to_remove:

        df = df.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

    # --------------------------------------------------------
    # PREPARE MARKET
    # --------------------------------------------------------

    market_copy = market.copy()

    market_copy["Date_Key"] = (
        market_copy["Date_Key"]
        .astype("int64")
    )

    market_copy = market_copy.sort_values(
        "Date_Key"
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = pd.merge_asof(
        df,
        market_copy,
        on="Date_Key",
        direction="backward"
    )

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    # --------------------------------------------------------

    df = add_relative_strength(
        df
    )

    # --------------------------------------------------------
    # REMOVE TEMPORARY KEY
    # --------------------------------------------------------

    df = df.drop(
        columns=["Date_Key"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        PROCESSED_DIR
        / f"{file_path.stem}{OUTPUT_SUFFIX}"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"  Saved: {output_file.name}"
    )

    print(
        f"  Rows: {len(df)}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "MARKET REGIME FEATURE ENGINEERING"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # ORIGINAL FEATURE FILES
    # --------------------------------------------------------

    feature_files = sorted(
        PROCESSED_DIR.glob(
            "*_features.csv"
        )
    )

    feature_files = [
        file
        for file in feature_files
        if not file.name.endswith(
            OUTPUT_SUFFIX
        )
    ]

    print(
        f"\nStock feature files found: "
        f"{len(feature_files)}"
    )

    if not feature_files:

        print(
            "No *_features.csv files found."
        )

        return

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    start_dates = []
    end_dates = []

    for file in feature_files:

        temp = pd.read_csv(
            file,
            usecols=["Date"]
        )

        temp["Date"] = pd.to_datetime(
            temp["Date"],
            errors="coerce"
        )

        start_dates.append(
            temp["Date"].min()
        )

        end_dates.append(
            temp["Date"].max()
        )

    start_date = (
        min(start_dates)
        - pd.Timedelta(days=300)
    )

    end_date = max(
        end_dates
    )

    # --------------------------------------------------------
    # NIFTY
    # --------------------------------------------------------

    nifty = download_nifty(
        start_date.strftime(
            "%Y-%m-%d"
        ),
        (
            end_date
            + pd.Timedelta(days=5)
        ).strftime(
            "%Y-%m-%d"
        )
    )

    market = build_market_features(
        nifty
    )

    print(
        f"NIFTY rows: {len(market)}"
    )

    # --------------------------------------------------------
    # REGIME DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nMarket regime distribution:"
    )

    print(
        market[
            "Market_Regime"
        ].value_counts()
    )

    # --------------------------------------------------------
    # PROCESS ALL STOCKS
    # --------------------------------------------------------

    successful = 0

    for file_path in feature_files:

        try:

            result = process_stock(
                file_path,
                market
            )

            if result:
                successful += 1

        except Exception as error:

            print(
                f"  ERROR: "
                f"{file_path.name}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("COMPLETE")
    print("=" * 65)

    print(
        f"Successful: "
        f"{successful}/{len(feature_files)}"
    )

    print(
        "\nOutput:"
    )

    print(
        "data/processed/*_features_regime.csv"
    )


if __name__ == "__main__":
    main()