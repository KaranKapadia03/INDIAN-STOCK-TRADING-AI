from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"


# ============================================================
# CONFIGURATION
# ============================================================

NEWS_LOOKBACK_DAYS = 7


NEWS_COLUMNS = [
    "News_Count",
    "News_Sentiment",
    "Positive_News_Count",
    "Negative_News_Count",
    "Neutral_News_Count",
    "News_Sentiment_Change",
    "News_Count_3D",
    "News_Count_7D",
    "News_Sentiment_3D",
    "News_Sentiment_7D",
]


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(filepath):

    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    print(f"Processing {symbol}...")

    # --------------------------------------------------------
    # Load stock feature data
    # --------------------------------------------------------

    stock = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    stock.sort_index(inplace=True)

    # Remove timezone from market dates
    stock.index = (
        pd.to_datetime(stock.index)
        .tz_localize(None)
        .normalize()
    )

    stock = stock.reset_index()

    date_column = stock.columns[0]

    stock.rename(
        columns={date_column: "date"},
        inplace=True
    )

    stock["date"] = (
        pd.to_datetime(stock["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    stock.sort_values(
        "date",
        inplace=True
    )

    # --------------------------------------------------------
    # Load news features
    # --------------------------------------------------------

    news_file = (
        NEWS_DIR
        / f"{symbol.replace('.', '_')}_news_features.csv"
    )

    if not news_file.exists():

        print(
            f"  WARNING: News file not found for {symbol}"
        )

        return None

    news = pd.read_csv(
        news_file
    )

    if news.empty:

        print(
            f"  WARNING: News file is empty for {symbol}"
        )

        return None

    # --------------------------------------------------------
    # Prepare news dates
    # --------------------------------------------------------

    news["date"] = (
        pd.to_datetime(
            news["date"],
            errors="coerce"
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    news.dropna(
        subset=["date"],
        inplace=True
    )

    news.sort_values(
        "date",
        inplace=True
    )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    available_columns = [
        column
        for column in NEWS_COLUMNS
        if column in news.columns
    ]

    news = news[
        ["date"] + available_columns
    ].copy()

    # --------------------------------------------------------
    # Force numeric news features
    # --------------------------------------------------------

    for column in available_columns:

        news[column] = pd.to_numeric(
            news[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove duplicate news dates
    # --------------------------------------------------------

    news = (
        news
        .groupby(
            "date",
            as_index=False
        )
        .mean(numeric_only=True)
    )

    news.sort_values(
        "date",
        inplace=True
    )

    # --------------------------------------------------------
    # Merge latest available news with each trading day
    # --------------------------------------------------------

    combined = pd.merge_asof(
        stock.sort_values("date"),
        news.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(
            days=NEWS_LOOKBACK_DAYS
        )
    )

    # --------------------------------------------------------
    # News availability
    # --------------------------------------------------------

    combined["News_Available"] = (
        combined["News_Sentiment"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # Days since latest news
    # --------------------------------------------------------

    news_dates = news[["date"]].copy()

    news_dates["News_Date"] = news_dates["date"]

    combined = pd.merge_asof(
        combined.sort_values("date"),
        news_dates[
            ["date", "News_Date"]
        ].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(
            days=NEWS_LOOKBACK_DAYS
        )
    )

    combined["Days_Since_News"] = (
        combined["date"]
        - combined["News_Date"]
    ).dt.days

    combined["Days_Since_News"] = (
        combined["Days_Since_News"]
        .fillna(-1)
        .astype(int)
    )

    combined.drop(
        columns=["News_Date"],
        inplace=True
    )

    # --------------------------------------------------------
    # Fill missing news values
    #
    # No recent news = neutral/no activity.
    # --------------------------------------------------------

    count_columns = [
        "News_Count",
        "Positive_News_Count",
        "Negative_News_Count",
        "Neutral_News_Count",
        "News_Count_3D",
        "News_Count_7D",
    ]

    for column in count_columns:

        if column in combined.columns:

            combined[column] = (
                combined[column]
                .fillna(0)
            )

    sentiment_columns = [
        "News_Sentiment",
        "News_Sentiment_Change",
        "News_Sentiment_3D",
        "News_Sentiment_7D",
    ]

    for column in sentiment_columns:

        if column in combined.columns:

            combined[column] = (
                combined[column]
                .fillna(0)
            )

    # --------------------------------------------------------
    # Restore index
    # --------------------------------------------------------

    combined.set_index(
        "date",
        inplace=True
    )

    combined.sort_index(
        inplace=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features_news.csv"
    )

    combined.to_csv(
        output_file
    )

    print(
        f"  Saved {len(combined)} rows, "
        f"{len(combined.columns)} columns"
    )

    return combined


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("TRADING-DAY NEWS FEATURE ENGINEERING")
    print("=" * 80)

    print(
        f"\nNews lookback window: "
        f"{NEWS_LOOKBACK_DAYS} calendar days"
    )

    files = sorted(
        FEATURES_DIR.glob("*_features.csv")
    )

    if not files:

        print("\nNo feature files found.")

        return

    successful = 0

    for filepath in files:

        try:

            result = process_stock(
                filepath
            )

            if result is not None:

                successful += 1

        except Exception as e:

            print(
                f"ERROR - {filepath.name}: {e}"
            )

    print("\n")
    print("=" * 80)
    print("TRADING-DAY NEWS FEATURES COMPLETE")
    print("=" * 80)

    print(
        f"\nSuccessfully processed: "
        f"{successful}/{len(files)}"
    )


if __name__ == "__main__":
    main()