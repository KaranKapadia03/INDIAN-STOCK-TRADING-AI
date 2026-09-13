from pathlib import Path

import pandas as pd

from news_sentiment import calculate_sentiment


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "news"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(filepath):

    print(f"Processing {filepath.name}...")

    df = pd.read_csv(filepath)

    if df.empty:
        print("  No news found.")
        return None

    # --------------------------------------------------------
    # Clean required columns
    # --------------------------------------------------------

    df["title"] = df["title"].fillna("")

    # --------------------------------------------------------
    # Calculate sentiment
    # --------------------------------------------------------

    sentiment_results = df["title"].apply(
        calculate_sentiment
    )

    sentiment_df = pd.DataFrame(
        sentiment_results.tolist()
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            sentiment_df.reset_index(drop=True)
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Publication date
    # --------------------------------------------------------

    df["published"] = pd.to_datetime(
        df["published"],
        errors="coerce",
        utc=True
    )

    df.dropna(
        subset=["published"],
        inplace=True
    )

    if df.empty:
        print("  No valid publication dates.")
        return None

    # --------------------------------------------------------
    # Convert timestamp to calendar date
    # --------------------------------------------------------

    df["date"] = (
        df["published"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = (
        df.groupby(
            ["date", "symbol", "company"],
            as_index=False
        )
        .agg(
            News_Count=("title", "count"),

            News_Sentiment=("score", "mean"),

            Positive_News_Count=(
                "label",
                lambda x: (x == "POSITIVE").sum()
            ),

            Negative_News_Count=(
                "label",
                lambda x: (x == "NEGATIVE").sum()
            ),

            Neutral_News_Count=(
                "label",
                lambda x: (x == "NEUTRAL").sum()
            ),
        )
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    daily.sort_values(
        "date",
        inplace=True
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    daily = daily.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------------
    # Force numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "News_Count",
        "News_Sentiment",
        "Positive_News_Count",
        "Negative_News_Count",
        "Neutral_News_Count",
    ]

    for column in numeric_columns:
        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Sentiment change
    # --------------------------------------------------------

    daily["News_Sentiment_Change"] = (
        daily["News_Sentiment"].diff()
    )

    # --------------------------------------------------------
    # Rolling news activity
    # --------------------------------------------------------

    daily["News_Count_3D"] = (
        daily["News_Count"].astype(float)
        .rolling(window=3, min_periods=1)
        .sum()
    )

    daily["News_Count_7D"] = (
        daily["News_Count"].astype(float)
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # --------------------------------------------------------
    # Rolling sentiment
    # --------------------------------------------------------

    daily["News_Sentiment_3D"] = (
        daily["News_Sentiment"].astype(float)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    daily["News_Sentiment_7D"] = (
        daily["News_Sentiment"].astype(float)
        .rolling(window=7, min_periods=1)
        .mean()
    )

# --------------------------------------------------------
# Rolling news activity
# --------------------------------------------------------

    daily["News_Count_3D"] = (
    daily["News_Count"]
    .rolling(
        window=3,
        min_periods=1
    )
    .sum()
)

    daily["News_Count_7D"] = (
    daily["News_Count"]
    .rolling(
        window=7,
        min_periods=1
    )
    .sum()
)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    symbol = daily["symbol"].iloc[0]

    output_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_news_features.csv"
    )

    daily.to_csv(
        output_file,
        index=False
    )

    print(
        f"  Saved {len(daily)} daily records → "
        f"{output_file.name}"
    )

    return daily


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("NEWS FEATURE ENGINEERING")
    print("=" * 80)

    files = sorted(
        NEWS_DIR.glob("*_news.csv")
    )

    if not files:

        print("\nNo news files found.")
        print("Run news_data.py first.")
        return

    successful = 0

    for filepath in files:

        try:

            result = process_stock(filepath)

            if result is not None:
                successful += 1

        except Exception as e:

            print(
                f"ERROR - {filepath.name}: {e}"
            )

    print("\n")
    print("=" * 80)
    print("NEWS FEATURE ENGINEERING COMPLETE")
    print("=" * 80)

    print(
        f"\nSuccessfully processed: "
        f"{successful}/{len(files)} stocks"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()