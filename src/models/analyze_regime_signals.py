from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "regime_walk_forward"
    / "regime_walk_forward_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "regime_signal_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("REGIME MODEL SIGNAL ANALYSIS")
print("=" * 70)

print("\nLoading predictions...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)

print(
    f"Predictions loaded: {len(df):,}"
)


# ============================================================
# BASIC INFO
# ============================================================

print("\n" + "=" * 70)
print("SIGNAL DISTRIBUTION")
print("=" * 70)

print(
    df["Signal"]
    .value_counts()
    .to_string()
)


# ============================================================
# BUY SIGNALS
# ============================================================

buy = df[
    df["Signal"] == "BUY"
].copy()

print("\n" + "=" * 70)
print("BUY PERFORMANCE")
print("=" * 70)

print(
    f"BUY signals: {len(buy):,}"
)

if len(buy) > 0:

    print(
        f"Average actual 20D return: "
        f"{buy['Actual_Return_20D'].mean():+.2f}%"
    )

    print(
        f"Median actual 20D return: "
        f"{buy['Actual_Return_20D'].median():+.2f}%"
    )

    print(
        f"Win rate: "
        f"{(
            buy["Actual_Return_20D"] > 0
        ).mean():.2%}"
    )

    print(
        f"+5% rate: "
        f"{(
            buy["Actual_Return_20D"] >= 5
        ).mean():.2%}"
    )

    print(
        f"-5% rate: "
        f"{(
            buy["Actual_Return_20D"] <= -5
        ).mean():.2%}"
    )


# ============================================================
# SELL SIGNALS
# ============================================================

sell = df[
    df["Signal"] == "SELL"
].copy()

print("\n" + "=" * 70)
print("SELL PERFORMANCE")
print("=" * 70)

print(
    f"SELL signals: {len(sell):,}"
)

if len(sell) > 0:

    print(
        f"Average actual 20D return: "
        f"{sell['Actual_Return_20D'].mean():+.2f}%"
    )

    print(
        f"Median actual 20D return: "
        f"{sell['Actual_Return_20D'].median():+.2f}%"
    )

    print(
        f"Correct SELL rate: "
        f"{(
            sell["Actual_Return_20D"] < 0
        ).mean():.2%}"
    )


# ============================================================
# BUY PROBABILITY BUCKETS
# ============================================================

print("\n" + "=" * 70)
print("BUY PROBABILITY BUCKETS")
print("=" * 70)

if len(buy) > 0:

    buy["Probability_Bucket"] = pd.cut(

        buy["Buy_Probability"],

        bins=[
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            1.01
        ],

        labels=[
            "0.60-0.65",
            "0.65-0.70",
            "0.70-0.75",
            "0.75-0.80",
            "0.80-0.85",
            "0.85-0.90",
            "0.90+"
        ],

        include_lowest=True
    )

    probability_analysis = (
        buy
        .groupby(
            "Probability_Bucket",
            observed=True
        )
        .agg(

            Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Actual_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Median_Actual_Return=(
                "Actual_Return_20D",
                "median"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Target_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x >= 5
                ).mean()
            ),

            Loss_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x <= -5
                ).mean()
            ),

            Avg_Expected_Return=(
                "Expected_Return_20D",
                "mean"
            )
        )
        .reset_index()
    )

    print(
        probability_analysis
        .to_string(index=False)
    )

    probability_analysis.to_csv(
        OUTPUT_DIR
        / "probability_buckets.csv",
        index=False
    )


# ============================================================
# BUY BY STOCK
# ============================================================

print("\n" + "=" * 70)
print("BUY PERFORMANCE BY STOCK")
print("=" * 70)

if len(buy) > 0:

    stock_analysis = (

        buy
        .groupby("Symbol")
        .agg(

            BUY_Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Actual_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Median_Actual_Return=(
                "Actual_Return_20D",
                "median"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Target_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x >= 5
                ).mean()
            ),

            Loss_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x <= -5
                ).mean()
            ),

            Avg_Buy_Probability=(
                "Buy_Probability",
                "mean"
            ),

            Avg_Sell_Probability=(
                "Sell_Probability",
                "mean"
            ),

            Avg_Expected_Return=(
                "Expected_Return_20D",
                "mean"
            ),

            Avg_Technical_Score=(
                "Technical_Score",
                "mean"
            ),

            Avg_Final_Score=(
                "Final_Score",
                "mean"
            )
        )

        .reset_index()

        .sort_values(
            "Avg_Actual_Return",
            ascending=False
        )
    )

    print(
        stock_analysis
        .to_string(index=False)
    )

    stock_analysis.to_csv(
        OUTPUT_DIR
        / "stock_analysis.csv",
        index=False
    )


# ============================================================
# BUY BY MARKET REGIME
# ============================================================

print("\n" + "=" * 70)
print("BUY PERFORMANCE BY MARKET REGIME")
print("=" * 70)

if len(buy) > 0:

    regime_analysis = (

        buy
        .groupby("Market_Regime")
        .agg(

            Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Actual_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Median_Return=(
                "Actual_Return_20D",
                "median"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Target_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x >= 5
                ).mean()
            ),

            Loss_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x <= -5
                ).mean()
            ),

            Avg_Buy_Probability=(
                "Buy_Probability",
                "mean"
            ),

            Avg_Expected_Return=(
                "Expected_Return_20D",
                "mean"
            )
        )

        .reset_index()
    )

    print(
        regime_analysis
        .to_string(index=False)
    )

    regime_analysis.to_csv(
        OUTPUT_DIR
        / "regime_analysis.csv",
        index=False
    )


# ============================================================
# FINAL SCORE BUCKETS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SCORE ANALYSIS")
print("=" * 70)

if len(buy) > 0:

    buy["Final_Score_Bucket"] = pd.cut(

        buy["Final_Score"],

        bins=[
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
            1.00
        ],

        labels=[
            "0.20-0.25",
            "0.25-0.30",
            "0.30-0.35",
            "0.35-0.40",
            "0.40-0.50",
            "0.50+"
        ],

        include_lowest=True
    )

    score_analysis = (

        buy
        .groupby(
            "Final_Score_Bucket",
            observed=True
        )
        .agg(

            Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Median_Return=(
                "Actual_Return_20D",
                "median"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Target_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x >= 5
                ).mean()
            )
        )

        .reset_index()
    )

    print(
        score_analysis
        .to_string(index=False)
    )

    score_analysis.to_csv(
        OUTPUT_DIR
        / "final_score_buckets.csv",
        index=False
    )


# ============================================================
# EXPECTED RETURN BUCKETS
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED RETURN ANALYSIS")
print("=" * 70)

if len(buy) > 0:

    buy["Expected_Return_Bucket"] = pd.cut(

        buy["Expected_Return_20D"],

        bins=[
            -np.inf,
            2,
            4,
            6,
            8,
            np.inf
        ],

        labels=[
            "<2%",
            "2-4%",
            "4-6%",
            "6-8%",
            "8%+"
        ]
    )

    expected_analysis = (

        buy
        .groupby(
            "Expected_Return_Bucket",
            observed=True
        )
        .agg(

            Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Actual_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Median_Return=(
                "Actual_Return_20D",
                "median"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Target_5pct_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x >= 5
                ).mean()
            )
        )

        .reset_index()
    )

    print(
        expected_analysis
        .to_string(index=False)
    )

    expected_analysis.to_csv(
        OUTPUT_DIR
        / "expected_return_buckets.csv",
        index=False
    )


# ============================================================
# STOCK + REGIME
# ============================================================

print("\n" + "=" * 70)
print("STOCK + REGIME ANALYSIS")
print("=" * 70)

if len(buy) > 0:

    stock_regime = (

        buy
        .groupby(
            [
                "Symbol",
                "Market_Regime"
            ]
        )
        .agg(

            Signals=(
                "Actual_Return_20D",
                "count"
            ),

            Avg_Return=(
                "Actual_Return_20D",
                "mean"
            ),

            Win_Rate=(
                "Actual_Return_20D",
                lambda x: (
                    x > 0
                ).mean()
            ),

            Avg_Buy_Probability=(
                "Buy_Probability",
                "mean"
            ),

            Avg_Final_Score=(
                "Final_Score",
                "mean"
            )
        )

        .reset_index()

        .sort_values(
            "Avg_Return",
            ascending=False
        )
    )

    print(
        stock_regime
        .to_string(index=False)
    )

    stock_regime.to_csv(
        OUTPUT_DIR
        / "stock_regime_analysis.csv",
        index=False
    )


# ============================================================
# SAVE OVERALL
# ============================================================

overall = pd.DataFrame({

    "Metric": [

        "Total Predictions",

        "BUY Signals",

        "SELL Signals",

        "BUY Avg Return",

        "BUY Win Rate",

        "BUY +5% Rate",

        "BUY -5% Rate",

        "SELL Avg Return",

        "SELL Correct Rate"
    ],

    "Value": [

        len(df),

        len(buy),

        len(sell),

        (
            buy["Actual_Return_20D"].mean()
            if len(buy)
            else np.nan
        ),

        (
            (
                buy["Actual_Return_20D"] > 0
            ).mean()
            if len(buy)
            else np.nan
        ),

        (
            (
                buy["Actual_Return_20D"] >= 5
            ).mean()
            if len(buy)
            else np.nan
        ),

        (
            (
                buy["Actual_Return_20D"] <= -5
            ).mean()
            if len(buy)
            else np.nan
        ),

        (
            sell["Actual_Return_20D"].mean()
            if len(sell)
            else np.nan
        ),

        (
            (
                sell["Actual_Return_20D"] < 0
            ).mean()
            if len(sell)
            else np.nan
        )
    ]
})


overall.to_csv(
    OUTPUT_DIR
    / "overall.csv",
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)

print("ANALYSIS COMPLETE")

print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_DIR}"
)