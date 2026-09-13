from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

RECENCY_DAYS = 7


# ============================================================
# CALCULATE LIVE NEWS SIGNAL
# ============================================================

def calculate_live_news_signal(filepath):

    symbol = (
        filepath.stem
        .replace("_news", "")
        .replace("_", ".")
    )

    data = pd.read_csv(
        filepath
    )

    if data.empty:
        return None

    # --------------------------------------------------------
    # Parse publication timestamps
    # --------------------------------------------------------

    data["published_timestamp"] = pd.to_datetime(
        data["published_timestamp"],
        errors="coerce",
        utc=True
    )

    data.dropna(
        subset=["published_timestamp"],
        inplace=True
    )

    if data.empty:
        return None

    # --------------------------------------------------------
    # Calculate article age
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    data["age_days"] = (
        now
        - data["published_timestamp"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # Keep recent articles
    # --------------------------------------------------------

    recent = data[
        (data["age_days"] >= 0)
        &
        (data["age_days"] <= RECENCY_DAYS)
    ].copy()

    if recent.empty:

        return {
            "symbol": symbol,
            "News_Count": 0,
            "News_Sentiment": 0.0,
            "Positive_News_Count": 0,
            "Negative_News_Count": 0,
            "Neutral_News_Count": 0,
            "News_Intensity": 0.0,
            "News_Recency_Score": 0.0,
            "News_Signal": 0.0,
            "News_Label": "NEUTRAL",
        }

    # --------------------------------------------------------
    # Import sentiment calculation
    # --------------------------------------------------------

    from news_sentiment import calculate_sentiment

    sentiment_results = (
        recent["title"]
        .fillna("")
        .apply(calculate_sentiment)
    )

    sentiment_df = pd.DataFrame(
        sentiment_results.tolist()
    )

    recent = pd.concat(
        [
            recent.reset_index(drop=True),
            sentiment_df.reset_index(drop=True)
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Basic sentiment
    # --------------------------------------------------------

    sentiment = recent["score"].mean()

    positive_count = (
        recent["label"] == "POSITIVE"
    ).sum()

    negative_count = (
        recent["label"] == "NEGATIVE"
    ).sum()

    neutral_count = (
        recent["label"] == "NEUTRAL"
    ).sum()

    article_count = len(recent)

    # --------------------------------------------------------
    # News intensity
    #
    # More news = stronger information flow.
    # Cap at 20 articles.
    # --------------------------------------------------------

    news_intensity = min(
        article_count / 20,
        1.0
    )

    # --------------------------------------------------------
    # Recency weighting
    #
    # Newer articles receive more weight.
    # --------------------------------------------------------

    recent["recency_weight"] = (
        1
        /
        (1 + recent["age_days"])
    )

    weighted_sentiment = (
        (
            recent["score"]
            * recent["recency_weight"]
        ).sum()
        /
        recent["recency_weight"].sum()
    )

    # --------------------------------------------------------
    # Recency score
    # --------------------------------------------------------

    recency_score = (
        recent["recency_weight"].mean()
    )

    # --------------------------------------------------------
    # Final news signal
    #
    # Sentiment contributes most.
    # Intensity strengthens the signal.
    # Recency strengthens fresh information.
    # --------------------------------------------------------

    news_signal = (
        weighted_sentiment
        * (
            0.60
            + 0.25 * news_intensity
            + 0.15 * recency_score
        )
    )

    news_signal = max(
        -1.0,
        min(1.0, news_signal)
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if news_signal >= 0.20:

        label = "POSITIVE"

    elif news_signal <= -0.20:

        label = "NEGATIVE"

    else:

        label = "NEUTRAL"

    return {
        "symbol": symbol,
        "News_Count": article_count,
        "News_Sentiment": sentiment,
        "Positive_News_Count": positive_count,
        "Negative_News_Count": negative_count,
        "Neutral_News_Count": neutral_count,
        "News_Intensity": news_intensity,
        "News_Recency_Score": recency_score,
        "News_Signal": news_signal,
        "News_Label": label,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("LIVE NEWS SIGNAL")
    print("=" * 80)

    files = sorted(
        NEWS_DIR.glob("*_news.csv")
    )

    if not files:

        print(
            "\nNo news files found."
        )

        return

    signals = []

    for filepath in files:

        try:

            result = (
                calculate_live_news_signal(
                    filepath
                )
            )

            if result is not None:

                signals.append(
                    result
                )

                print(
                    f"{result['symbol']:15s} | "
                    f"Articles: "
                    f"{result['News_Count']:3d} | "
                    f"Sentiment: "
                    f"{result['News_Sentiment']:+.3f} | "
                    f"Signal: "
                    f"{result['News_Signal']:+.3f} | "
                    f"{result['News_Label']}"
                )

        except Exception as e:

            print(
                f"ERROR - "
                f"{filepath.name}: {e}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if signals:

        result_df = pd.DataFrame(
            signals
        )

        result_df.sort_values(
            "News_Signal",
            ascending=False,
            inplace=True
        )

        output_file = (
            OUTPUT_DIR
            / "live_news_signals.csv"
        )

        result_df.to_csv(
            output_file,
            index=False
        )

        print("\n")
        print("=" * 80)
        print("LIVE NEWS SIGNAL COMPLETE")
        print("=" * 80)

        print(
            f"\nStocks processed: "
            f"{len(result_df)}"
        )

        print(
            f"\nSaved to:\n"
            f"{output_file}"
        )

        print("\n")
        print(
            result_df.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()