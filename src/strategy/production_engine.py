from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

LIVE_ML_FILE = (
    FEATURE_DIR
    / "live_ml"
    / "live_predictions.csv"
)

NEWS_FILE = (
    FEATURE_DIR
    / "news"
    / "live_news_signals.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "production"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "recommendations.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "recommendations.json"
)


# ============================================================
# WEIGHTS
# ============================================================

ML_WEIGHT = 0.50
TECHNICAL_WEIGHT = 0.25
NEWS_WEIGHT = 0.25


# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

BUY_SCORE = 0.20
SELL_SCORE = -0.20

BUY_EXPECTED_RETURN = 2.0
SELL_EXPECTED_RETURN = -2.0


# ============================================================
# LOAD LIVE ML
# ============================================================

def load_ml_predictions():

    if not LIVE_ML_FILE.exists():

        raise FileNotFoundError(
            f"\nLive ML predictions not found:\n"
            f"{LIVE_ML_FILE}\n\n"
            "Run first:\n"
            "python src\\models\\live_predictions.py"
        )

    df = pd.read_csv(
        LIVE_ML_FILE
    )

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    if df.empty:

        raise RuntimeError(
            "Live ML prediction file is empty."
        )

    required = [
        "Symbol",
        "Prediction",
        "ML_Normalized",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing ML columns: {missing}"
        )

    return df


# ============================================================
# LOAD LATEST TECHNICAL DATA
# ============================================================

def load_latest_features():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    rows = []

    for file in files:

        df = pd.read_csv(
            file
        )

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ]

        if df.empty:
            continue

        if "Date" not in df.columns:
            continue

        if "Close" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date"]
        )

        if df.empty:
            continue

        df = df.sort_values(
            "Date"
        )

        row = (
            df.iloc[-1]
            .copy()
        )

        symbol = (
            file.stem
            .replace(
                "_features",
                ""
            )
            .replace(
                "_NS",
                ".NS"
            )
        )

        row["Symbol"] = symbol

        rows.append(
            row
        )

    if not rows:

        raise RuntimeError(
            "No technical feature files found."
        )

    result = pd.DataFrame(
        rows
    )

    result = result.loc[
        :,
        ~result.columns.duplicated()
    ]

    return result


# ============================================================
# LOAD NEWS
# ============================================================

def load_news():

    if not NEWS_FILE.exists():

        print(
            "\nWARNING: Live news file not found."
        )

        return pd.DataFrame(
            columns=[
                "Symbol",
                "News_Sentiment",
                "News_Count",
            ]
        )

    news = pd.read_csv(
        NEWS_FILE
    )

    news = news.loc[
        :,
        ~news.columns.duplicated()
    ]

    return news


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(
    row
):

    score = 0

    reasons = []

    # Price vs SMA20

    if (
        "Close" in row
        and "SMA_20" in row
        and pd.notna(row["SMA_20"])
    ):

        if row["Close"] > row["SMA_20"]:

            score += 1

        else:

            score -= 1

    # SMA20 vs SMA50

    if (
        "SMA_20" in row
        and "SMA_50" in row
        and pd.notna(row["SMA_20"])
        and pd.notna(row["SMA_50"])
    ):

        if row["SMA_20"] > row["SMA_50"]:

            score += 1

        else:

            score -= 1

    # MACD

    if (
        "MACD" in row
        and "MACD_Signal" in row
        and pd.notna(row["MACD"])
        and pd.notna(row["MACD_Signal"])
    ):

        if row["MACD"] > row["MACD_Signal"]:

            score += 1

        else:

            score -= 1

    # RSI

    if (
        "RSI_14" in row
        and pd.notna(row["RSI_14"])
    ):

        rsi = float(
            row["RSI_14"]
        )

        if 50 <= rsi <= 70:

            score += 1

        elif rsi < 30:

            score += 1

        elif rsi > 70:

            score -= 1

    # Bollinger position

    if (
        "BB_Position" in row
        and pd.notna(row["BB_Position"])
    ):

        bb = float(
            row["BB_Position"]
        )

        if bb > 0.5:

            score += 1

        elif bb < 0.2:

            score -= 1

    normalized = (
        score / 5.0
    )

    if score >= 3:

        reasons.append(
            "Strong positive technical trend"
        )

    elif score <= -3:

        reasons.append(
            "Strong negative technical trend"
        )

    elif score > 0:

        reasons.append(
            "Positive technical signals"
        )

    elif score < 0:

        reasons.append(
            "Negative technical signals"
        )

    else:

        reasons.append(
            "Mixed technical signals"
        )

    return (
        score,
        normalized,
        reasons
    )


# ============================================================
# NEWS SCORE
# ============================================================

def calculate_news_score(
    news_row
):

    if news_row is None:

        return (
            0.0,
            "No recent news data"
        )

    sentiment = 0.0

    if (
        "News_Sentiment"
        in news_row
    ):

        value = pd.to_numeric(
            news_row[
                "News_Sentiment"
            ],
            errors="coerce"
        )

        if pd.notna(value):

            sentiment = float(
                value
            )

    # Clamp unexpected values.

    sentiment = np.clip(
        sentiment,
        -1.0,
        1.0
    )

    if sentiment >= 0.15:

        label = "Positive recent news"

    elif sentiment <= -0.15:

        label = "Negative recent news"

    else:

        label = "Neutral recent news"

    return (
        sentiment,
        label
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_signal(
    final_score,
    expected_return,
    ml_score
):

    if (
        final_score >= BUY_SCORE
        and expected_return >= BUY_EXPECTED_RETURN
        and ml_score > 0
    ):

        return "BUY"

    if (
        final_score <= SELL_SCORE
        and expected_return <= SELL_EXPECTED_RETURN
        and ml_score < 0
    ):

        return "SELL"

    return "HOLD"


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    final_score,
    expected_return,
    ml_score,
    technical_score,
    news_score
):

    score_strength = min(
        abs(final_score),
        1.0
    )

    expected_strength = min(
        abs(expected_return) / 10.0,
        1.0
    )

    ml_strength = min(
        abs(ml_score),
        1.0
    )

    technical_strength = (
        abs(technical_score) / 5.0
    )

    news_strength = min(
        abs(news_score),
        1.0
    )

    confidence = (
        score_strength * 0.40
        + expected_strength * 0.20
        + ml_strength * 0.20
        + technical_strength * 0.10
        + news_strength * 0.10
    )

    return round(
        confidence * 100,
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INDIAN STOCK TRADING AI"
    )

    print(
        "INTEGRATED AI RECOMMENDATION ENGINE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\nLoading live ML predictions..."
    )

    ml = load_ml_predictions()

    print(
        f"ML stocks: "
        f"{len(ml)}"
    )

    print(
        "\nLoading technical features..."
    )

    technical = (
        load_latest_features()
    )

    print(
        f"Technical stocks: "
        f"{len(technical)}"
    )

    print(
        "\nLoading live news..."
    )

    news = load_news()

    print(
        f"News rows: "
        f"{len(news)}"
    )

    # --------------------------------------------------------
    # MERGE ML + TECHNICAL
    # --------------------------------------------------------

    df = pd.merge(
        ml,
        technical,
        on="Symbol",
        how="left",
        suffixes=(
            "_ML",
            "_TECH"
        )
    )

    # --------------------------------------------------------
    # MERGE NEWS
    # --------------------------------------------------------

    if not news.empty:

        news = news.loc[
            :,
            ~news.columns.duplicated()
        ]

        # Normalize symbol column.

        if "symbol" in news.columns:

            news = news.rename(
                columns={
                    "symbol": "Symbol"
                }
            )

        if "Symbol" in news.columns:

            news_subset = news[
                [
                    c
                    for c in [
                        "Symbol",
                        "News_Sentiment",
                        "News_Count",
                        "News_Signal",
                        "News_Label",
                    ]
                    if c in news.columns
                ]
            ].copy()

            df = pd.merge(
                df,
                news_subset,
                on="Symbol",
                how="left"
            )

    # --------------------------------------------------------
    # CALCULATE SIGNALS
    # --------------------------------------------------------

    recommendations = []

    for _, row in df.iterrows():

        symbol = row[
            "Symbol"
        ]

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        ml_score = float(
            row.get(
                "ML_Normalized",
                0
            )
        )

        prediction = float(
            row.get(
                "Prediction",
                0
            )
        )

        # The model predicts excess return.
        # Add the NIFTY expected component to make
        # the displayed expected return a stock-return
        # estimate.

        nifty_return = float(
            row.get(
                "NIFTY_Return_20D",
                0
            )
        )

        expected_return = (
            prediction
            + nifty_return
        )

        # ----------------------------------------------------
        # TECHNICAL
        # ----------------------------------------------------

        (
            technical_raw,
            technical_score,
            technical_reasons
        ) = calculate_technical_score(
            row
        )

        # ----------------------------------------------------
        # NEWS
        # ----------------------------------------------------

        news_score, news_reason = (
            calculate_news_score(
                row
            )
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (
            ml_score * ML_WEIGHT
            + technical_score
            * TECHNICAL_WEIGHT
            + news_score
            * NEWS_WEIGHT
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal = generate_signal(
            final_score,
            expected_return,
            ml_score
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = (
            calculate_confidence(
                final_score,
                expected_return,
                ml_score,
                technical_raw,
                news_score
            )
        )

        # ----------------------------------------------------
        # ML REASON
        # ----------------------------------------------------

        if ml_score >= 0.50:

            ml_reason = (
                "Strong positive ML ranking"
            )

        elif ml_score >= 0.10:

            ml_reason = (
                "Positive ML ranking"
            )

        elif ml_score <= -0.50:

            ml_reason = (
                "Strong negative ML ranking"
            )

        elif ml_score < -0.10:

            ml_reason = (
                "Negative ML ranking"
            )

        else:

            ml_reason = (
                "Neutral ML ranking"
            )

        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = (
            technical_reasons
            + [
                ml_reason,
                news_reason
            ]
        )

        recommendations.append(
            {
                "Symbol": symbol,

                "Date": row.get(
                    "Date_ML",
                    row.get("Date")
                ),

                "Close": row.get(
                    "Close_ML",
                    row.get("Close")
                ),

                "Signal": signal,

                "Expected_Return": round(
                    expected_return,
                    3
                ),

                "ML_Prediction": round(
                    prediction,
                    3
                ),

                "ML_Score": round(
                    ml_score,
                    4
                ),

                "Technical_Score": int(
                    technical_raw
                ),

                "Technical_Normalized": round(
                    technical_score,
                    4
                ),

                "News_Score": round(
                    news_score,
                    4
                ),

                "Final_Score": round(
                    final_score,
                    4
                ),

                "Confidence": confidence,

                "Reasons": "; ".join(
                    reasons
                ),
            }
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        recommendations
    )

    result = result.sort_values(
        [
            "Signal",
            "Final_Score"
        ],
        ascending=[
            True,
            False
        ]
    )

    # Put BUY first, then HOLD, then SELL.

    signal_order = {
        "BUY": 0,
        "HOLD": 1,
        "SELL": 2,
    }

    result[
        "_order"
    ] = result[
        "Signal"
    ].map(
        signal_order
    )

    result = result.sort_values(
        [
            "_order",
            "Final_Score"
        ],
        ascending=[
            True,
            False
        ]
    )

    result = result.drop(
        columns=["_order"]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    result.to_json(
        OUTPUT_JSON,
        orient="records",
        indent=2
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "FINAL AI RECOMMENDATIONS"
    )

    print("=" * 70)

    display_columns = [
        "Symbol",
        "Close",
        "Signal",
        "Expected_Return",
        "Confidence",
        "Final_Score",
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)

    print(
        "SIGNAL SUMMARY"
    )

    print("=" * 70)

    print(
        result[
            "Signal"
        ].value_counts()
        .to_string()
    )

    print("\n")
    print("=" * 70)

    print(
        "FILES SAVED"
    )

    print("=" * 70)

    print(
        f"\nCSV:\n"
        f"{OUTPUT_CSV.resolve()}"
    )

    print(
        f"\nJSON:\n"
        f"{OUTPUT_JSON.resolve()}"
    )

    print("\n")
    print(
        "INTEGRATED RECOMMENDATION ENGINE COMPLETE"
    )


if __name__ == "__main__":

    main()