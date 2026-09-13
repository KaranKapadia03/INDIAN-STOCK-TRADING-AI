from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "walk_forward_portfolio"
    / "walk_forward_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "prediction_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    numeric_columns = [
        "ML_Probability",
        "ML_Score",
        "Technical_Score",
        "News_Score",
        "Final_Score",
        "Future_Return_5D",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df.dropna(
        subset=[
            "ML_Probability",
            "Future_Return_5D"
        ],
        inplace=True
    )

    return df


# ============================================================
# PROBABILITY BUCKET ANALYSIS
# ============================================================

def probability_analysis(df):

    bins = [
        0.0,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        1.01,
    ]

    labels = [
        "<50%",
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-85%",
        "85%+",
    ]

    df["Probability_Bucket"] = pd.cut(
        df["ML_Probability"],
        bins=bins,
        labels=labels,
        right=False
    )

    grouped = (
        df.groupby(
            "Probability_Bucket",
            observed=False
        )
        .agg(
            observations=(
                "Future_Return_5D",
                "count"
            ),
            average_return_pct=(
                "Future_Return_5D",
                "mean"
            ),
            median_return_pct=(
                "Future_Return_5D",
                "median"
            ),
            win_rate_pct=(
                "Future_Return_5D",
                lambda x:
                    (x > 0).mean() * 100
            ),
            return_std_pct=(
                "Future_Return_5D",
                "std"
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# STOCK-LEVEL ANALYSIS
# ============================================================

def stock_analysis(df):

    result = (
        df.groupby("symbol")
        .agg(
            observations=(
                "Future_Return_5D",
                "count"
            ),
            average_future_return_pct=(
                "Future_Return_5D",
                "mean"
            ),
            median_future_return_pct=(
                "Future_Return_5D",
                "median"
            ),
            actual_win_rate_pct=(
                "Future_Return_5D",
                lambda x:
                    (x > 0).mean() * 100
            ),
            average_ml_probability=(
                "ML_Probability",
                "mean"
            ),
            ml_probability_std=(
                "ML_Probability",
                "std"
            ),
            average_final_score=(
                "Final_Score",
                "mean"
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# HIGH-CONFIDENCE ANALYSIS
# ============================================================

def threshold_analysis(df):

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    rows = []

    for threshold in thresholds:

        subset = df[
            df["ML_Probability"]
            >= threshold
        ]

        if subset.empty:

            continue

        rows.append(
            {
                "threshold":
                    threshold,

                "observations":
                    len(subset),

                "average_return_pct":
                    subset[
                        "Future_Return_5D"
                    ].mean(),

                "median_return_pct":
                    subset[
                        "Future_Return_5D"
                    ].median(),

                "win_rate_pct":
                    (
                        subset[
                            "Future_Return_5D"
                        ] > 0
                    ).mean() * 100,

                "average_probability":
                    subset[
                        "ML_Probability"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    columns = [
        "ML_Probability",
        "ML_Score",
        "Technical_Score",
        "News_Score",
        "Final_Score",
        "Future_Return_5D",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    correlation = (
        df[available]
        .corr()
    )

    return correlation


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def signal_analysis(df):

    result = []

    # --------------------------------------------------------
    # ML BUY
    # --------------------------------------------------------

    ml_buy = df[
        df["ML_Probability"]
        >= 0.60
    ]

    result.append(
        {
            "signal":
                "ML >= 60%",
            "observations":
                len(ml_buy),
            "average_return_pct":
                ml_buy[
                    "Future_Return_5D"
                ].mean(),
            "win_rate_pct":
                (
                    ml_buy[
                        "Future_Return_5D"
                    ] > 0
                ).mean() * 100,
        }
    )

    # --------------------------------------------------------
    # ML + TECHNICAL
    # --------------------------------------------------------

    ml_technical = df[
        (
            df["ML_Probability"]
            >= 0.60
        )
        &
        (
            df["Technical_Score"]
            > 0
        )
    ]

    result.append(
        {
            "signal":
                "ML >= 60% + Technical > 0",
            "observations":
                len(ml_technical),
            "average_return_pct":
                ml_technical[
                    "Future_Return_5D"
                ].mean(),
            "win_rate_pct":
                (
                    ml_technical[
                        "Future_Return_5D"
                    ] > 0
                ).mean() * 100,
        }
    )

    # --------------------------------------------------------
    # FULL SIGNAL
    # --------------------------------------------------------

    full = df[
        (
            df["ML_Probability"]
            >= 0.60
        )
        &
        (
            df["Final_Score"]
            > 0.15
        )
    ]

    result.append(
        {
            "signal":
                "Full BUY Signal",
            "observations":
                len(full),
            "average_return_pct":
                full[
                    "Future_Return_5D"
                ].mean(),
            "win_rate_pct":
                (
                    full[
                        "Future_Return_5D"
                    ] > 0
                ).mean() * 100,
        }
    )

    return pd.DataFrame(
        result
    )


# ============================================================
# BEST / WORST STOCKS
# ============================================================

def best_worst_stocks(
    stock_results
):

    best = (
        stock_results
        .sort_values(
            "average_future_return_pct",
            ascending=False
        )
        .head(5)
    )

    worst = (
        stock_results
        .sort_values(
            "average_future_return_pct",
            ascending=True
        )
        .head(5)
    )

    return best, worst


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "ML PREDICTION ANALYSIS"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_predictions()

    print(
        f"\nTotal predictions: "
        f"{len(df):,}"
    )

    print(
        f"Stocks: "
        f"{df['symbol'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Probability buckets
    # --------------------------------------------------------

    probability = probability_analysis(
        df
    )

    probability_file = (
        OUTPUT_DIR
        / "probability_buckets.csv"
    )

    probability.to_csv(
        probability_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print(
        "PROBABILITY BUCKETS"
    )
    print("=" * 80)

    print(
        probability.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    thresholds = threshold_analysis(
        df
    )

    threshold_file = (
        OUTPUT_DIR
        / "threshold_analysis.csv"
    )

    thresholds.to_csv(
        threshold_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print(
        "PROBABILITY THRESHOLDS"
    )
    print("=" * 80)

    print(
        thresholds.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stock analysis
    # --------------------------------------------------------

    stocks = stock_analysis(
        df
    )

    stocks_file = (
        OUTPUT_DIR
        / "stock_analysis.csv"
    )

    stocks.to_csv(
        stocks_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print(
        "STOCK-LEVEL PERFORMANCE"
    )
    print("=" * 80)

    print(
        stocks.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best / worst
    # --------------------------------------------------------

    best, worst = (
        best_worst_stocks(
            stocks
        )
    )

    print("\n")
    print("=" * 80)
    print(
        "BEST STOCKS"
    )
    print("=" * 80)

    print(
        best.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print(
        "WORST STOCKS"
    )
    print("=" * 80)

    print(
        worst.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation = (
        correlation_analysis(
            df
        )
    )

    correlation_file = (
        OUTPUT_DIR
        / "correlations.csv"
    )

    correlation.to_csv(
        correlation_file
    )

    print("\n")
    print("=" * 80)
    print(
        "CORRELATION WITH FUTURE RETURN"
    )
    print("=" * 80)

    if "Future_Return_5D" in correlation.columns:

        print(
            correlation[
                "Future_Return_5D"
            ]
            .sort_values(
                ascending=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # Signal analysis
    # --------------------------------------------------------

    signals = signal_analysis(
        df
    )

    signals_file = (
        OUTPUT_DIR
        / "signal_analysis.csv"
    )

    signals.to_csv(
        signals_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print(
        "SIGNAL ANALYSIS"
    )
    print("=" * 80)

    print(
        signals.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Overall prediction statistics
    # --------------------------------------------------------

    probability_correlation = (
        df[
            "ML_Probability"
        ]
        .corr(
            df[
                "Future_Return_5D"
            ]
        )
    )

    average_return = (
        df[
            "Future_Return_5D"
        ].mean()
    )

    actual_win_rate = (
        df[
            "Future_Return_5D"
        ]
        .gt(0)
        .mean()
        * 100
    )

    print("\n")
    print("=" * 80)
    print(
        "OVERALL MODEL DIAGNOSTICS"
    )
    print("=" * 80)

    print(
        f"\nProbability / future-return correlation: "
        f"{probability_correlation:.4f}"
    )

    print(
        f"Average 5D future return: "
        f"{average_return:.4f}%"
    )

    print(
        f"Overall actual UP rate: "
        f"{actual_win_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "predictions":
                    len(df),

                "stocks":
                    df["symbol"].nunique(),

                "probability_return_correlation":
                    probability_correlation,

                "average_future_return_pct":
                    average_return,

                "actual_up_rate_pct":
                    actual_win_rate,
            }
        ]
    )

    summary_file = (
        OUTPUT_DIR
        / "model_diagnostics.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print(
        "PREDICTION ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nSaved analysis to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()