from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "v11_logistic_model"
    / "v11_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "v12_threshold_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Thresholds to test
THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V12 - PROBABILITY CALIBRATION & THRESHOLD ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    f"Stocks: {df['Symbol'].nunique()}"
)

print(
    f"Date range: "
    f"{df['Date'].min().date()} "
    f"-> "
    f"{df['Date'].max().date()}"
)


# ============================================================
# BASIC DATA
# ============================================================

df["Buy_Probability"] = pd.to_numeric(
    df["Buy_Probability"],
    errors="coerce"
)

df["Actual_20D_Return"] = pd.to_numeric(
    df["Actual_20D_Return"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Buy_Probability",
        "Actual_20D_Return"
    ]
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

threshold_results = []


for threshold in THRESHOLDS:

    signals = df[
        df["Buy_Probability"]
        >= threshold
    ].copy()

    if signals.empty:
        continue

    returns = signals[
        "Actual_20D_Return"
    ]

    threshold_results.append(
        {
            "Threshold":
                threshold,

            "Signals":
                len(signals),

            "Signal_Percentage":
                len(signals) / len(df) * 100,

            "Average_Return":
                returns.mean(),

            "Median_Return":
                returns.median(),

            "Win_Rate":
                (returns > 0).mean() * 100,

            "Return_Above_2":
                (returns >= 2).mean() * 100,

            "Return_Above_5":
                (returns >= 5).mean() * 100,

            "Return_Below_Minus_2":
                (returns <= -2).mean() * 100,

            "Return_Below_Minus_5":
                (returns <= -5).mean() * 100,

            "Best_Return":
                returns.max(),

            "Worst_Return":
                returns.min(),

            "Return_Std":
                returns.std(),
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("BUY PROBABILITY THRESHOLDS")
print("=" * 70)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.2f}"
    )
)


# ============================================================
# PROBABILITY BUCKETS
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY BUCKET ANALYSIS")
print("=" * 70)


bins = [
    0.00,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    1.01,
]

labels = [
    "<0.50",
    "0.50-0.55",
    "0.55-0.60",
    "0.60-0.65",
    "0.65-0.70",
    "0.70-0.75",
    "0.75-0.80",
    "0.80-0.85",
    "0.85-0.90",
    "0.90+",
]


df["Probability_Bucket"] = pd.cut(
    df["Buy_Probability"],
    bins=bins,
    labels=labels,
    right=False
)


bucket_results = []

for bucket, group in df.groupby(
    "Probability_Bucket",
    observed=False
):

    if len(group) == 0:
        continue

    returns = group[
        "Actual_20D_Return"
    ]

    bucket_results.append(
        {
            "Probability_Bucket":
                str(bucket),

            "Signals":
                len(group),

            "Average_Return":
                returns.mean(),

            "Median_Return":
                returns.median(),

            "Win_Rate":
                (returns > 0).mean() * 100,

            "Above_5_Rate":
                (returns >= 5).mean() * 100,

            "Below_Minus_5_Rate":
                (returns <= -5).mean() * 100,
        }
    )


bucket_df = pd.DataFrame(
    bucket_results
)

print(
    bucket_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.2f}"
    )
)


# ============================================================
# STOCK-LEVEL THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("STOCK-LEVEL ANALYSIS")
print("=" * 70)


stock_results = []

for symbol, stock in df.groupby(
    "Symbol"
):

    for threshold in [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]:

        signals = stock[
            stock["Buy_Probability"]
            >= threshold
        ]

        if len(signals) < 5:
            continue

        returns = signals[
            "Actual_20D_Return"
        ]

        stock_results.append(
            {
                "Symbol":
                    symbol,

                "Threshold":
                    threshold,

                "Signals":
                    len(signals),

                "Average_Return":
                    returns.mean(),

                "Win_Rate":
                    (returns > 0).mean()
                    * 100,

                "Above_5_Rate":
                    (returns >= 5).mean()
                    * 100,

                "Below_Minus_5_Rate":
                    (returns <= -5).mean()
                    * 100,
            }
        )


stock_df = pd.DataFrame(
    stock_results
)

if not stock_df.empty:

    print(
        stock_df
        .sort_values(
            "Average_Return",
            ascending=False
        )
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )


# ============================================================
# DATE / PERIOD STABILITY
# ============================================================

print("\n" + "=" * 70)
print("PERIOD STABILITY")
print("=" * 70)


df["Year"] = (
    df["Date"]
    .dt.year
)


period_results = []


for year, group in df.groupby(
    "Year"
):

    signals = group[
        group["Buy_Probability"]
        >= 0.60
    ]

    if len(signals) == 0:
        continue

    returns = signals[
        "Actual_20D_Return"
    ]

    period_results.append(
        {
            "Year":
                year,

            "Signals":
                len(signals),

            "Average_Return":
                returns.mean(),

            "Median_Return":
                returns.median(),

            "Win_Rate":
                (returns > 0).mean()
                * 100,

            "Above_5_Rate":
                (returns >= 5).mean()
                * 100,

            "Below_Minus_5_Rate":
                (returns <= -5).mean()
                * 100,
        }
    )


period_df = pd.DataFrame(
    period_results
)

print(
    period_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.2f}"
    )
)


# ============================================================
# TOP STOCKS
# ============================================================

print("\n" + "=" * 70)
print("BEST STOCKS AT >= 0.60 BUY PROBABILITY")
print("=" * 70)


signals_60 = df[
    df["Buy_Probability"]
    >= 0.60
]


best_stocks = (
    signals_60
    .groupby("Symbol")
    .agg(
        Signals=(
            "Actual_20D_Return",
            "count"
        ),

        Average_Return=(
            "Actual_20D_Return",
            "mean"
        ),

        Median_Return=(
            "Actual_20D_Return",
            "median"
        ),

        Win_Rate=(
            "Actual_20D_Return",
            lambda x:
            (x > 0).mean() * 100
        ),

        Above_5_Rate=(
            "Actual_20D_Return",
            lambda x:
            (x >= 5).mean() * 100
        ),
    )
    .sort_values(
        "Average_Return",
        ascending=False
    )
)


print(
    best_stocks.to_string(
        float_format=lambda x:
        f"{x:.2f}"
    )
)


# ============================================================
# CALIBRATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION CHECK")
print("=" * 70)


calibration_results = []


for bucket, group in df.groupby(
    "Probability_Bucket",
    observed=False
):

    if len(group) < 10:
        continue

    predicted_probability = (
        group["Buy_Probability"]
        .mean()
    )

    actual_win_rate = (
        group["Actual_20D_Return"]
        > 0
    ).mean()

    calibration_results.append(
        {
            "Bucket":
                str(bucket),

            "Average_Predicted_Probability":
                predicted_probability,

            "Actual_Win_Rate":
                actual_win_rate,

            "Calibration_Error":
                actual_win_rate
                - predicted_probability,

            "Signals":
                len(group),
        }
    )


calibration_df = pd.DataFrame(
    calibration_results
)

print(
    calibration_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

threshold_df.to_csv(
    OUTPUT_DIR
    / "threshold_analysis.csv",
    index=False
)

bucket_df.to_csv(
    OUTPUT_DIR
    / "probability_buckets.csv",
    index=False
)

stock_df.to_csv(
    OUTPUT_DIR
    / "stock_threshold_analysis.csv",
    index=False
)

period_df.to_csv(
    OUTPUT_DIR
    / "period_stability.csv",
    index=False
)

best_stocks.to_csv(
    OUTPUT_DIR
    / "best_stocks.csv"
)

calibration_df.to_csv(
    OUTPUT_DIR
    / "calibration.csv",
    index=False
)


print("\n" + "=" * 70)
print("V12 COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_DIR)
