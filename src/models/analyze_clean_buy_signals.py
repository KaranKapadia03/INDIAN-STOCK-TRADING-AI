from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = (
    Path("data")
    / "models"
    / "true_clean_walk_forward"
    / "clean_walk_forward_predictions.csv"
)

OUTPUT_DIR = (
    Path("data")
    / "models"
    / "clean_buy_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(PREDICTIONS_FILE)

    # Standardize column names
    df = df.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "buy_probability": "Buy_Probability",
            "sell_probability": "Sell_Probability",
            "expected_return_20d": "Expected_Return",
            "technical_score": "Technical_Score",
            "final_score": "Final_Score",
            "signal": "Signal",
            "actual_return_20d": "Actual_Return",
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Buy_Probability",
        "Sell_Probability",
        "Expected_Return",
        "Technical_Score",
        "Final_Score",
        "Actual_Return",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Symbol",
            "Date",
            "Buy_Probability",
            "Expected_Return",
            "Actual_Return",
        ]
    )

    return df


# ============================================================
# BASIC BUY ANALYSIS
# ============================================================

def analyze_all_buys(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    if buys.empty:
        return pd.DataFrame()

    buys["Win"] = (
        buys["Actual_Return"] > 0
    )

    buys["Target_Hit"] = (
        buys["Actual_Return"] >= 5
    )

    buys["Loss_5pct"] = (
        buys["Actual_Return"] <= -5
    )

    summary = pd.DataFrame([{
        "BUY_Signals": len(buys),
        "Average_Actual_Return": buys["Actual_Return"].mean(),
        "Median_Actual_Return": buys["Actual_Return"].median(),
        "Win_Rate": buys["Win"].mean(),
        "Target_5pct_Rate": buys["Target_Hit"].mean(),
        "Loss_5pct_Rate": buys["Loss_5pct"].mean(),
        "Average_Buy_Probability": buys["Buy_Probability"].mean(),
        "Average_Expected_Return": buys["Expected_Return"].mean(),
        "Average_Final_Score": buys["Final_Score"].mean(),
    }])

    return summary


# ============================================================
# PROBABILITY BUCKET ANALYSIS
# ============================================================

def analyze_probability_buckets(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    if buys.empty:
        return pd.DataFrame()

    bins = [
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
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90+",
    ]

    buys["Probability_Bucket"] = pd.cut(
        buys["Buy_Probability"],
        bins=bins,
        labels=labels,
        right=False
    )

    result = (
        buys
        .groupby(
            "Probability_Bucket",
            observed=False
        )
        .agg(
            Signals=("Actual_Return", "count"),
            Avg_Actual_Return=("Actual_Return", "mean"),
            Median_Actual_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_5pct_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
            Loss_5pct_Rate=(
                "Actual_Return",
                lambda x: (x <= -5).mean()
            ),
            Avg_Expected_Return=(
                "Expected_Return",
                "mean"
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    thresholds = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    rows = []

    for threshold in thresholds:

        subset = buys[
            buys["Buy_Probability"] >= threshold
        ]

        if subset.empty:
            continue

        actual = subset["Actual_Return"]

        rows.append({
            "Probability_Threshold": threshold,
            "Signals": len(subset),
            "Average_Return": actual.mean(),
            "Median_Return": actual.median(),
            "Win_Rate": (actual > 0).mean(),
            "Target_5pct_Rate": (actual >= 5).mean(),
            "Loss_5pct_Rate": (actual <= -5).mean(),
            "Average_Expected_Return": (
                subset["Expected_Return"].mean()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stocks(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    if buys.empty:
        return pd.DataFrame()

    result = (
        buys
        .groupby("Symbol")
        .agg(
            BUY_Signals=("Actual_Return", "count"),
            Avg_Actual_Return=("Actual_Return", "mean"),
            Median_Actual_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_5pct_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
            Loss_5pct_Rate=(
                "Actual_Return",
                lambda x: (x <= -5).mean()
            ),
            Avg_Buy_Probability=(
                "Buy_Probability",
                "mean"
            ),
            Avg_Expected_Return=(
                "Expected_Return",
                "mean"
            ),
            Avg_Technical_Score=(
                "Technical_Score",
                "mean"
            ),
        )
        .reset_index()
    )

    result = result.sort_values(
        "Avg_Actual_Return",
        ascending=False
    )

    return result


# ============================================================
# EXPECTED RETURN ANALYSIS
# ============================================================

def analyze_expected_return(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    bins = [
        -np.inf,
        0,
        2,
        4,
        6,
        8,
        np.inf,
    ]

    labels = [
        "<0%",
        "0-2%",
        "2-4%",
        "4-6%",
        "6-8%",
        "8%+",
    ]

    buys["Expected_Return_Bucket"] = pd.cut(
        buys["Expected_Return"],
        bins=bins,
        labels=labels,
        right=False
    )

    result = (
        buys
        .groupby(
            "Expected_Return_Bucket",
            observed=False
        )
        .agg(
            Signals=("Actual_Return", "count"),
            Avg_Actual_Return=("Actual_Return", "mean"),
            Median_Actual_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_5pct_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# COMBINED FILTER ANALYSIS
# ============================================================

def analyze_filters(df):

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ].copy()

    filters = [
        {
            "Name": "BUY >= 0.60",
            "condition": buys["Buy_Probability"] >= 0.60,
        },
        {
            "Name": "BUY >= 0.65",
            "condition": buys["Buy_Probability"] >= 0.65,
        },
        {
            "Name": "BUY >= 0.70",
            "condition": buys["Buy_Probability"] >= 0.70,
        },
        {
            "Name": "BUY >= 0.75",
            "condition": buys["Buy_Probability"] >= 0.75,
        },
        {
            "Name": "BUY >= 0.80",
            "condition": buys["Buy_Probability"] >= 0.80,
        },
        {
            "Name": "BUY >= 0.70 AND Expected >= 5",
            "condition": (
                (buys["Buy_Probability"] >= 0.70)
                & (buys["Expected_Return"] >= 5)
            ),
        },
        {
            "Name": "BUY >= 0.75 AND Expected >= 5",
            "condition": (
                (buys["Buy_Probability"] >= 0.75)
                & (buys["Expected_Return"] >= 5)
            ),
        },
        {
            "Name": "BUY >= 0.80 AND Expected >= 5",
            "condition": (
                (buys["Buy_Probability"] >= 0.80)
                & (buys["Expected_Return"] >= 5)
            ),
        },
    ]

    rows = []

    for item in filters:

        subset = buys[item["condition"]]

        if subset.empty:
            rows.append({
                "Filter": item["Name"],
                "Signals": 0,
                "Avg_Return": np.nan,
                "Median_Return": np.nan,
                "Win_Rate": np.nan,
                "Target_5pct_Rate": np.nan,
            })

            continue

        actual = subset["Actual_Return"]

        rows.append({
            "Filter": item["Name"],
            "Signals": len(subset),
            "Avg_Return": actual.mean(),
            "Median_Return": actual.median(),
            "Win_Rate": (actual > 0).mean(),
            "Target_5pct_Rate": (actual >= 5).mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("CLEAN BUY SIGNAL ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"\nTotal predictions: {len(df)}"
    )

    buys = df[
        df["Signal"].str.upper() == "BUY"
    ]

    print(
        f"Total BUY signals: {len(buys)}"
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = analyze_all_buys(df)

    print("\n")
    print("=" * 70)
    print("OVERALL BUY PERFORMANCE")
    print("=" * 70)

    if not overall.empty:

        row = overall.iloc[0]

        print(
            f"BUY signals             : "
            f"{int(row['BUY_Signals'])}"
        )

        print(
            f"Average actual return   : "
            f"{row['Average_Actual_Return']:+.2f}%"
        )

        print(
            f"Median actual return    : "
            f"{row['Median_Actual_Return']:+.2f}%"
        )

        print(
            f"Win rate                : "
            f"{row['Win_Rate'] * 100:.2f}%"
        )

        print(
            f"+5% target rate         : "
            f"{row['Target_5pct_Rate'] * 100:.2f}%"
        )

        print(
            f"-5% loss rate           : "
            f"{row['Loss_5pct_Rate'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability = analyze_probability_buckets(df)

    print("\n")
    print("=" * 70)
    print("BUY PROBABILITY BUCKETS")
    print("=" * 70)

    print(
        probability.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # THRESHOLDS
    # --------------------------------------------------------

    thresholds = analyze_thresholds(df)

    print("\n")
    print("=" * 70)
    print("PROBABILITY THRESHOLDS")
    print("=" * 70)

    print(
        thresholds.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # STOCKS
    # --------------------------------------------------------

    stocks = analyze_stocks(df)

    print("\n")
    print("=" * 70)
    print("BUY PERFORMANCE BY STOCK")
    print("=" * 70)

    print(
        stocks.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # EXPECTED RETURN
    # --------------------------------------------------------

    expected = analyze_expected_return(df)

    print("\n")
    print("=" * 70)
    print("EXPECTED RETURN BUCKETS")
    print("=" * 70)

    print(
        expected.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filters = analyze_filters(df)

    print("\n")
    print("=" * 70)
    print("COMBINED FILTER ANALYSIS")
    print("=" * 70)

    print(
        filters.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    overall.to_csv(
        OUTPUT_DIR / "overall_buy_performance.csv",
        index=False
    )

    probability.to_csv(
        OUTPUT_DIR / "probability_buckets.csv",
        index=False
    )

    thresholds.to_csv(
        OUTPUT_DIR / "probability_thresholds.csv",
        index=False
    )

    stocks.to_csv(
        OUTPUT_DIR / "stock_analysis.csv",
        index=False
    )

    expected.to_csv(
        OUTPUT_DIR / "expected_return_buckets.csv",
        index=False
    )

    filters.to_csv(
        OUTPUT_DIR / "filter_analysis.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        OUTPUT_DIR / "overall_buy_performance.csv"
    )

    print(
        OUTPUT_DIR / "probability_buckets.csv"
    )

    print(
        OUTPUT_DIR / "probability_thresholds.csv"
    )

    print(
        OUTPUT_DIR / "stock_analysis.csv"
    )

    print(
        OUTPUT_DIR / "expected_return_buckets.csv"
    )

    print(
        OUTPUT_DIR / "filter_analysis.csv"
    )


if __name__ == "__main__":
    main()