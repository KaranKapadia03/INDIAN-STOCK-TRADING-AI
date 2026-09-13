from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEWS_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
)

SENTIMENT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news"
)

SENTIMENT_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Simple financial sentiment dictionary
# --------------------------------------------------

POSITIVE_WORDS = {
    "growth",
    "profit",
    "profits",
    "surge",
    "surges",
    "rally",
    "strong",
    "stronger",
    "positive",
    "gain",
    "gains",
    "increase",
    "increased",
    "improved",
    "improvement",
    "record",
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "bullish",
    "rise",
    "rises",
    "rising",
    "success",
    "successful",
    "expansion",
    "expand",
    "expands",
    "partnership",
    "contract",
    "order",
    "orders",
    "revenue",
    "earnings",
    "optimistic",
    "outperform",
    "outperformance",
}


NEGATIVE_WORDS = {
    "loss",
    "losses",
    "fall",
    "falls",
    "falling",
    "drop",
    "drops",
    "decline",
    "declined",
    "declining",
    "weak",
    "weaker",
    "negative",
    "risk",
    "risks",
    "warning",
    "downgrade",
    "downgraded",
    "bearish",
    "debt",
    "fraud",
    "scandal",
    "lawsuit",
    "penalty",
    "fine",
    "investigation",
    "probe",
    "crisis",
    "concern",
    "concerns",
    "cut",
    "cuts",
    "slump",
    "slumps",
    "miss",
    "misses",
    "missed",
    "underperform",
    "underperformance",
}


def clean_text(text):
    """
    Convert headline into lowercase words.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    words = text.split()

    return words


def calculate_sentiment(text):
    """
    Calculate simple financial sentiment.

    Score range:

    -1 = strongly negative
     0 = neutral
    +1 = strongly positive
    """

    words = clean_text(text)

    positive_count = sum(
        word in POSITIVE_WORDS
        for word in words
    )

    negative_count = sum(
        word in NEGATIVE_WORDS
        for word in words
    )

    total_sentiment_words = (
        positive_count
        + negative_count
    )

    if total_sentiment_words == 0:

        score = 0.0

    else:

        score = (
            positive_count
            - negative_count
        ) / total_sentiment_words

    if score > 0.20:

        label = "POSITIVE"

    elif score < -0.20:

        label = "NEGATIVE"

    else:

        label = "NEUTRAL"

    return {
        "score": score,
        "label": label,
        "positive_words": positive_count,
        "negative_words": negative_count,
    }


def analyze_company_news(symbol):
    """
    Analyze all collected news for one company.
    """

    filename = (
        symbol.replace(".", "_")
        + "_news.csv"
    )

    filepath = (
        NEWS_DATA_DIR / filename
    )

    if not filepath.exists():

        raise FileNotFoundError(
            f"News file not found: {filepath}"
        )

    data = pd.read_csv(
        filepath
    )

    if data.empty:

        return pd.DataFrame()

    # --------------------------------------------------
    # Calculate sentiment for each headline
    # --------------------------------------------------

    sentiment_results = data[
        "title"
    ].apply(
        calculate_sentiment
    )

    sentiment_df = pd.DataFrame(
        sentiment_results.tolist()
    )

    data = pd.concat(
        [
            data.reset_index(drop=True),
            sentiment_df
        ],
        axis=1
    )

    return data


def create_company_summary(
    symbol,
    data
):
    """
    Create one aggregated sentiment score
    for a company.
    """

    if data.empty:

        return None

    average_score = data[
        "score"
    ].mean()

    positive_articles = (
        data["label"] == "POSITIVE"
    ).sum()

    negative_articles = (
        data["label"] == "NEGATIVE"
    ).sum()

    neutral_articles = (
        data["label"] == "NEUTRAL"
    ).sum()

    total_articles = len(data)

    return {
        "symbol": symbol,
        "articles": total_articles,
        "average_sentiment":
            average_score,
        "positive_articles":
            positive_articles,
        "negative_articles":
            negative_articles,
        "neutral_articles":
            neutral_articles,
    }


def save_sentiment(
    data,
    symbol
):
    """
    Save article-level sentiment.
    """

    if data.empty:

        return None

    filename = (
        symbol.replace(".", "_")
        + "_sentiment.csv"
    )

    filepath = (
        SENTIMENT_DATA_DIR / filename
    )

    data.to_csv(
        filepath,
        index=False
    )

    return filepath


def main():

    print("\n" + "=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "NEWS SENTIMENT ANALYSIS"
    )
    print("=" * 80)

    summaries = []

    news_files = sorted(
        NEWS_DATA_DIR.glob(
            "*_news.csv"
        )
    )

    if not news_files:

        print(
            "\nNo news files found."
        )

        print(
            "Run news_data.py first."
        )

        return

    for filepath in news_files:

        symbol = (
            filepath.stem
            .replace(
                "_news",
                ""
            )
            .replace(
                "_",
                "."
            )
        )

        try:

            print(
                f"\nAnalyzing {symbol}..."
            )

            data = analyze_company_news(
                symbol
            )

            save_sentiment(
                data,
                symbol
            )

            summary = (
                create_company_summary(
                    symbol,
                    data
                )
            )

            if summary:

                summaries.append(
                    summary
                )

                print(
                    f"Articles: "
                    f"{summary['articles']}"
                )

                print(
                    f"Average Sentiment: "
                    f"{summary['average_sentiment']:.3f}"
                )

                print(
                    f"Positive: "
                    f"{summary['positive_articles']}"
                )

                print(
                    f"Negative: "
                    f"{summary['negative_articles']}"
                )

                print(
                    f"Neutral: "
                    f"{summary['neutral_articles']}"
                )

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

    # --------------------------------------------------
    # Save company-level summary
    # --------------------------------------------------

    if summaries:

        summary_df = pd.DataFrame(
            summaries
        )

        summary_df.sort_values(
            "average_sentiment",
            ascending=False,
            inplace=True
        )

        output_file = (
            SENTIMENT_DATA_DIR
            / "company_sentiment_summary.csv"
        )

        summary_df.to_csv(
            output_file,
            index=False
        )

        print("\n" + "=" * 80)
        print(
            "NEWS SENTIMENT COMPLETE"
        )
        print("=" * 80)

        print(
            summary_df.to_string(
                index=False
            )
        )

        print(
            f"\nSaved to:\n"
            f"{output_file}"
        )


if __name__ == "__main__":
    main()