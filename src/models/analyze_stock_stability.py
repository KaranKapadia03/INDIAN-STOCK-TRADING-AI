from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_FILE = (
    Path("data")
    / "models"
    / "true_clean_walk_forward"
    / "clean_walk_forward_predictions.csv"
)

OUTPUT_DIR = (
    Path("data")
    / "models"
    / "stock_stability_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    df = pd.read_csv(PREDICTIONS_FILE)

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

    df["Signal"] = (
        df["Signal"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "Symbol",
            "Date",
            "Buy_Probability",
            "Actual_Return",
        ]
    )

    return df


def add_periods(df):

    df = df.copy()

    df["Year"] = df["Date"].dt.year

    df["Half"] = np.where(
        df["Date"].dt.month <= 6,
        "H1",
        "H2"
    )

    df["Year_Half"] = (
        df["Year"].astype(str)
        + "_"
        + df["Half"]
    )

    df["Quarter"] = (
        df["Date"]
        .dt.to_period("Q")
        .astype(str)
    )

    return df


def yearly_analysis(df):

    buys = df[df["Signal"] == "BUY"].copy()

    if buys.empty:
        return pd.DataFrame()

    result = (
        buys
        .groupby(["Symbol", "Year"])
        .agg(
            Signals=("Actual_Return", "count"),
            Avg_Return=("Actual_Return", "mean"),
            Median_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
            Loss_Rate=(
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
        )
        .reset_index()
    )

    return result


def half_year_analysis(df):

    buys = df[df["Signal"] == "BUY"].copy()

    if buys.empty:
        return pd.DataFrame()

    result = (
        buys
        .groupby(["Symbol", "Year_Half"])
        .agg(
            Signals=("Actual_Return", "count"),
            Avg_Return=("Actual_Return", "mean"),
            Median_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
            Loss_Rate=(
                "Actual_Return",
                lambda x: (x <= -5).mean()
            ),
        )
        .reset_index()
    )

    return result


def consistency_analysis(df):

    buys = df[df["Signal"] == "BUY"].copy()

    rows = []

    for symbol, group in buys.groupby("Symbol"):

        if len(group) < 5:
            continue

        yearly = (
            group
            .groupby("Year")
            .agg(
                Avg_Return=("Actual_Return", "mean"),
                Signals=("Actual_Return", "count")
            )
        )

        profitable_years = (
            yearly["Avg_Return"] > 0
        ).sum()

        total_years = len(yearly)

        avg_return = group["Actual_Return"].mean()
        median_return = group["Actual_Return"].median()
        return_std = group["Actual_Return"].std()
        win_rate = (
            group["Actual_Return"] > 0
        ).mean()

        year_consistency = (
            profitable_years / total_years
            if total_years > 0
            else 0
        )

        rows.append(
            {
                "Symbol": symbol,
                "BUY_Signals": len(group),
                "Average_Return": avg_return,
                "Median_Return": median_return,
                "Return_Std": return_std,
                "Win_Rate": win_rate,
                "Profitable_Years": profitable_years,
                "Total_Years": total_years,
                "Year_Consistency": year_consistency,
                "Avg_Buy_Probability": (
                    group["Buy_Probability"].mean()
                ),
                "Avg_Expected_Return": (
                    group["Expected_Return"].mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["Stability_Score"] = (
        result["Win_Rate"] * 0.40
        + result["Year_Consistency"] * 0.40
        + (
            result["Average_Return"] > 0
        ).astype(float) * 0.20
    )

    result = result.sort_values(
        ["Stability_Score", "Average_Return"],
        ascending=False
    )

    return result


def high_confidence_analysis(df):

    buys = df[
        (df["Signal"] == "BUY")
        & (df["Buy_Probability"] >= 0.70)
    ].copy()

    if buys.empty:
        return pd.DataFrame()

    result = (
        buys
        .groupby("Symbol")
        .agg(
            Signals=("Actual_Return", "count"),
            Avg_Return=("Actual_Return", "mean"),
            Median_Return=("Actual_Return", "median"),
            Win_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            ),
            Target_Rate=(
                "Actual_Return",
                lambda x: (x >= 5).mean()
            ),
            Loss_Rate=(
                "Actual_Return",
                lambda x: (x <= -5).mean()
            ),
            Avg_Probability=(
                "Buy_Probability",
                "mean"
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "Avg_Return",
        ascending=False
    )


def rank_stocks(stability):

    if stability.empty:
        return stability

    result = stability.copy()

    result["Eligible"] = (
        result["BUY_Signals"] >= 10
    )

    result["Rank_Score"] = (
        result["Average_Return"] * 0.40
        + result["Win_Rate"] * 10 * 0.25
        + result["Year_Consistency"] * 10 * 0.25
        + result["Stability_Score"] * 10 * 0.10
    )

    result = result.sort_values(
        "Rank_Score",
        ascending=False
    )

    return result


def main():

    print("\n")
    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("CLEAN STOCK STABILITY ANALYSIS")
    print("=" * 70)

    df = load_data()
    df = add_periods(df)

    print(
        f"\nTotal predictions: {len(df)}"
    )

    print(
        f"BUY signals: "
        f"{(df['Signal'] == 'BUY').sum()}"
    )

    print(
        f"Stocks: {df['Symbol'].nunique()}"
    )

    yearly = yearly_analysis(df)

    print("\n")
    print("=" * 70)
    print("YEARLY BUY PERFORMANCE")
    print("=" * 70)

    if yearly.empty:
        print("No BUY data.")
    else:
        print(
            yearly.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    half_year = half_year_analysis(df)

    print("\n")
    print("=" * 70)
    print("HALF-YEAR BUY PERFORMANCE")
    print("=" * 70)

    if half_year.empty:
        print("No BUY data.")
    else:
        print(
            half_year.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    stability = consistency_analysis(df)

    print("\n")
    print("=" * 70)
    print("STOCK CONSISTENCY")
    print("=" * 70)

    if stability.empty:
        print("No stability data.")
    else:
        print(
            stability.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    ranked = rank_stocks(stability)

    print("\n")
    print("=" * 70)
    print("STOCK RANKING")
    print("=" * 70)

    if ranked.empty:
        print("No ranking data.")
    else:
        ranking_columns = [
            "Symbol",
            "BUY_Signals",
            "Average_Return",
            "Win_Rate",
            "Year_Consistency",
            "Stability_Score",
            "Rank_Score",
            "Eligible",
        ]

        print(
            ranked[ranking_columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    high_confidence = high_confidence_analysis(df)

    print("\n")
    print("=" * 70)
    print("HIGH-CONFIDENCE BUY PERFORMANCE")
    print("BUY PROBABILITY >= 0.70")
    print("=" * 70)

    if high_confidence.empty:
        print("No high-confidence BUY data.")
    else:
        print(
            high_confidence.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    yearly.to_csv(
        OUTPUT_DIR / "yearly_buy_performance.csv",
        index=False
    )

    half_year.to_csv(
        OUTPUT_DIR / "half_year_buy_performance.csv",
        index=False
    )

    stability.to_csv(
        OUTPUT_DIR / "stock_consistency.csv",
        index=False
    )

    ranked.to_csv(
        OUTPUT_DIR / "stock_ranking.csv",
        index=False
    )

    high_confidence.to_csv(
        OUTPUT_DIR / "high_confidence_buy.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        OUTPUT_DIR / "yearly_buy_performance.csv"
    )

    print(
        OUTPUT_DIR / "half_year_buy_performance.csv"
    )

    print(
        OUTPUT_DIR / "stock_consistency.csv"
    )

    print(
        OUTPUT_DIR / "stock_ranking.csv"
    )

    print(
        OUTPUT_DIR / "high_confidence_buy.csv"
    )


if __name__ == "__main__":
    main()