from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRADE_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "strategy_v8_validation"
    / "all_period_trades.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "strategy_v9_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):

    columns = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in columns:
            return columns[candidate.lower()]

    for column in df.columns:

        column_lower = str(column).lower().strip()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


def print_section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V9 - SIGNAL LEVEL ANALYSIS")
print("=" * 70)

print("\nLoading V8 trades...")

if not TRADE_FILE.exists():

    raise FileNotFoundError(
        f"\nV8 trade file not found:\n{TRADE_FILE}\n"
        "Run V8 first."
    )

df = pd.read_csv(TRADE_FILE)

print(
    f"Trades loaded: {len(df):,}"
)

print(
    f"Columns: {list(df.columns)}"
)


# ============================================================
# COLUMN DETECTION
# ============================================================

symbol_col = find_column(
    df,
    ["Symbol", "Ticker", "Stock"]
)

return_col = find_column(
    df,
    ["Return", "Trade_Return"]
)

pnl_col = find_column(
    df,
    ["Net_PnL", "PnL", "Profit"]
)

buy_prob_col = find_column(
    df,
    ["Buy_Probability", "BuyProbability"]
)

expected_col = find_column(
    df,
    [
        "Expected_Return_20D",
        "Expected_Return",
    ]
)

technical_col = find_column(
    df,
    [
        "Technical_Score",
        "TechnicalScore",
    ]
)

regime_col = find_column(
    df,
    [
        "Market_Regime",
        "Regime",
    ]
)

rank_col = find_column(
    df,
    [
        "Rank_Score",
        "RankScore",
    ]
)

period_col = find_column(
    df,
    ["Period"]
)

days_col = find_column(
    df,
    [
        "Days_Held",
        "Holding_Days",
    ]
)

exit_col = find_column(
    df,
    [
        "Exit_Reason",
        "ExitReason",
    ]
)


required = {
    "Symbol": symbol_col,
    "Return": return_col,
    "PnL": pnl_col,
    "Buy Probability": buy_prob_col,
    "Expected Return": expected_col,
    "Technical Score": technical_col,
    "Regime": regime_col,
    "Rank Score": rank_col,
    "Period": period_col,
}

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# CLEAN
# ============================================================

df[return_col] = pd.to_numeric(
    df[return_col],
    errors="coerce"
)

df[pnl_col] = pd.to_numeric(
    df[pnl_col],
    errors="coerce"
)

df[buy_prob_col] = pd.to_numeric(
    df[buy_prob_col],
    errors="coerce"
)

df[expected_col] = pd.to_numeric(
    df[expected_col],
    errors="coerce"
)

df[technical_col] = pd.to_numeric(
    df[technical_col],
    errors="coerce"
)

df[rank_col] = pd.to_numeric(
    df[rank_col],
    errors="coerce"
)

df[period_col] = pd.to_numeric(
    df[period_col],
    errors="coerce"
)

if days_col is not None:

    df[days_col] = pd.to_numeric(
        df[days_col],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        return_col,
        pnl_col,
        buy_prob_col,
        expected_col,
        technical_col,
        rank_col,
        period_col,
    ]
).copy()


# ============================================================
# NORMALIZE RETURN
# ============================================================

if df[return_col].abs().median() > 1:

    df["Trade_Return"] = (
        df[return_col] / 100
    )

else:

    df["Trade_Return"] = (
        df[return_col]
    )

df["Win"] = (
    df["Trade_Return"] > 0
)


# ============================================================
# PERIOD ANALYSIS
# ============================================================

print_section(
    "SIGNAL PERFORMANCE BY PERIOD"
)

period_stats = (
    df.groupby(period_col)
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
        Average_Buy_Probability=(
            buy_prob_col,
            "mean"
        ),
        Average_Expected_Return=(
            expected_col,
            "mean"
        ),
        Average_Technical_Score=(
            technical_col,
            "mean"
        ),
        Average_Rank_Score=(
            rank_col,
            "mean"
        ),
    )
    .sort_index()
)

print(
    period_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

period_stats.to_csv(
    OUTPUT_DIR / "period_signal_stats.csv"
)


# ============================================================
# BUY PROBABILITY ANALYSIS
# ============================================================

print_section(
    "BUY PROBABILITY ANALYSIS"
)

df["Buy_Probability_Bucket"] = pd.cut(
    df[buy_prob_col],
    bins=[
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        1.00,
    ],
    include_lowest=True,
)

prob_stats = (
    df.groupby(
        "Buy_Probability_Bucket",
        observed=False
    )
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
        Average_Expected_Return=(
            expected_col,
            "mean"
        ),
    )
)

print(
    prob_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

prob_stats.to_csv(
    OUTPUT_DIR / "buy_probability_analysis.csv"
)


# ============================================================
# EXPECTED RETURN ANALYSIS
# ============================================================

print_section(
    "EXPECTED RETURN ANALYSIS"
)

df["Expected_Return_Bucket"] = pd.cut(
    df[expected_col],
    bins=[
        -100,
        0,
        2,
        4,
        6,
        8,
        10,
        15,
        100,
    ],
    include_lowest=True,
)

expected_stats = (
    df.groupby(
        "Expected_Return_Bucket",
        observed=False
    )
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
    )
)

print(
    expected_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

expected_stats.to_csv(
    OUTPUT_DIR / "expected_return_analysis.csv"
)


# ============================================================
# TECHNICAL SCORE ANALYSIS
# ============================================================

print_section(
    "TECHNICAL SCORE ANALYSIS"
)

technical_stats = (
    df.groupby(technical_col)
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
    )
    .sort_index()
)

print(
    technical_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

technical_stats.to_csv(
    OUTPUT_DIR / "technical_score_analysis.csv"
)


# ============================================================
# RANK SCORE ANALYSIS
# ============================================================

print_section(
    "RANK SCORE ANALYSIS"
)

df["Rank_Bucket"] = pd.cut(
    df[rank_col],
    bins=[
        -100,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        1.00,
    ],
    include_lowest=True,
)

rank_stats = (
    df.groupby(
        "Rank_Bucket",
        observed=False
    )
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
    )
)

print(
    rank_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

rank_stats.to_csv(
    OUTPUT_DIR / "rank_score_analysis.csv"
)


# ============================================================
# STOCK ANALYSIS
# ============================================================

print_section(
    "STOCK PERFORMANCE ACROSS ALL PERIODS"
)

stock_stats = (
    df.groupby(symbol_col)
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
        Average_Buy_Probability=(
            buy_prob_col,
            "mean"
        ),
        Average_Expected_Return=(
            expected_col,
            "mean"
        ),
        Average_Rank=(
            rank_col,
            "mean"
        ),
    )
    .sort_values(
        "Total_PnL",
        ascending=False
    )
)

print(
    stock_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

stock_stats.to_csv(
    OUTPUT_DIR / "stock_analysis.csv"
)


# ============================================================
# REGIME ANALYSIS
# ============================================================

print_section(
    "REGIME + SIGNAL ANALYSIS"
)

regime_stats = (
    df.groupby(regime_col)
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Median_Return=("Trade_Return", "median"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
        Average_Buy_Probability=(
            buy_prob_col,
            "mean"
        ),
        Average_Expected_Return=(
            expected_col,
            "mean"
        ),
        Average_Technical_Score=(
            technical_col,
            "mean"
        ),
        Average_Rank=(
            rank_col,
            "mean"
        ),
    )
    .sort_values(
        "Total_PnL",
        ascending=False
    )
)

print(
    regime_stats.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

regime_stats.to_csv(
    OUTPUT_DIR / "regime_analysis.csv"
)


# ============================================================
# PERIOD × REGIME
# ============================================================

print_section(
    "PERIOD × REGIME"
)

period_regime = (
    df.groupby(
        [
            period_col,
            regime_col,
        ]
    )
    .agg(
        Trades=("Trade_Return", "count"),
        Average_Return=("Trade_Return", "mean"),
        Win_Rate=("Win", "mean"),
        Total_PnL=(pnl_col, "sum"),
    )
    .reset_index()
)

print(
    period_regime.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)

period_regime.to_csv(
    OUTPUT_DIR / "period_regime_analysis.csv",
    index=False
)


# ============================================================
# SIGNAL QUALITY TEST
# ============================================================

print_section(
    "HIGH-CONVICTION SIGNAL TEST"
)

tests = [
    (
        "Buy Prob >= 0.55",
        df[buy_prob_col] >= 0.55,
    ),
    (
        "Buy Prob >= 0.60",
        df[buy_prob_col] >= 0.60,
    ),
    (
        "Buy Prob >= 0.65",
        df[buy_prob_col] >= 0.65,
    ),
    (
        "Buy Prob >= 0.70",
        df[buy_prob_col] >= 0.70,
    ),
    (
        "Buy Prob >= 0.75",
        df[buy_prob_col] >= 0.75,
    ),
    (
        "Expected >= 2%",
        df[expected_col] >= 2,
    ),
    (
        "Expected >= 4%",
        df[expected_col] >= 4,
    ),
    (
        "Expected >= 6%",
        df[expected_col] >= 6,
    ),
    (
        "Expected >= 8%",
        df[expected_col] >= 8,
    ),
    (
        "Technical >= 1",
        df[technical_col] >= 1,
    ),
    (
        "Technical >= 2",
        df[technical_col] >= 2,
    ),
    (
        "Rank >= 0.30",
        df[rank_col] >= 0.30,
    ),
    (
        "Rank >= 0.40",
        df[rank_col] >= 0.40,
    ),
]

test_rows = []

for name, mask in tests:

    subset = df.loc[mask]

    if subset.empty:
        continue

    test_rows.append(
        {
            "Condition": name,
            "Trades": len(subset),
            "Average_Return":
                subset["Trade_Return"].mean(),
            "Median_Return":
                subset["Trade_Return"].median(),
            "Win_Rate":
                subset["Win"].mean(),
            "Total_PnL":
                subset[pnl_col].sum(),
        }
    )

test_df = pd.DataFrame(test_rows)

print(
    test_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)

test_df.to_csv(
    OUTPUT_DIR / "high_conviction_tests.csv",
    index=False
)


# ============================================================
# COMBINATION TESTS
# ============================================================

print_section(
    "SIGNAL COMBINATION TESTS"
)

combination_tests = {

    "BuyProb>=0.60 & Expected>=2":
        (
            (df[buy_prob_col] >= 0.60)
            &
            (df[expected_col] >= 2)
        ),

    "BuyProb>=0.65 & Expected>=2":
        (
            (df[buy_prob_col] >= 0.65)
            &
            (df[expected_col] >= 2)
        ),

    "BuyProb>=0.65 & Expected>=4":
        (
            (df[buy_prob_col] >= 0.65)
            &
            (df[expected_col] >= 4)
        ),

    "BuyProb>=0.70 & Expected>=4":
        (
            (df[buy_prob_col] >= 0.70)
            &
            (df[expected_col] >= 4)
        ),

    "BuyProb>=0.70 & Technical>=1":
        (
            (df[buy_prob_col] >= 0.70)
            &
            (df[technical_col] >= 1)
        ),

    "BuyProb>=0.70 & Expected>=4 & Technical>=1":
        (
            (df[buy_prob_col] >= 0.70)
            &
            (df[expected_col] >= 4)
            &
            (df[technical_col] >= 1)
        ),

    "Rank>=0.30 & Expected>=4":
        (
            (df[rank_col] >= 0.30)
            &
            (df[expected_col] >= 4)
        ),

    "Rank>=0.40 & Expected>=4":
        (
            (df[rank_col] >= 0.40)
            &
            (df[expected_col] >= 4)
        ),
}

combination_rows = []

for name, mask in combination_tests.items():

    subset = df.loc[mask]

    if subset.empty:
        continue

    combination_rows.append(
        {
            "Condition": name,
            "Trades": len(subset),
            "Average_Return":
                subset["Trade_Return"].mean(),
            "Median_Return":
                subset["Trade_Return"].median(),
            "Win_Rate":
                subset["Win"].mean(),
            "Total_PnL":
                subset[pnl_col].sum(),
        }
    )

combination_df = pd.DataFrame(
    combination_rows
)

print(
    combination_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)

combination_df.to_csv(
    OUTPUT_DIR / "combination_tests.csv",
    index=False
)


# ============================================================
# HOLDING PERIOD
# ============================================================

if days_col is not None:

    print_section(
        "HOLDING PERIOD ANALYSIS"
    )

    df["Holding_Bucket"] = pd.cut(
        df[days_col],
        bins=[
            0,
            5,
            10,
            15,
            20,
            100,
        ],
        include_lowest=True
    )

    holding_stats = (
        df.groupby(
            "Holding_Bucket",
            observed=False
        )
        .agg(
            Trades=("Trade_Return", "count"),
            Average_Return=(
                "Trade_Return",
                "mean"
            ),
            Median_Return=(
                "Trade_Return",
                "median"
            ),
            Win_Rate=("Win", "mean"),
            Total_PnL=(pnl_col, "sum"),
        )
    )

    print(
        holding_stats.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    holding_stats.to_csv(
        OUTPUT_DIR / "holding_period_analysis.csv"
    )


# ============================================================
# CORRELATIONS
# ============================================================

print_section(
    "SIGNAL CORRELATIONS"
)

correlation_columns = [
    buy_prob_col,
    expected_col,
    technical_col,
    rank_col,
    "Trade_Return",
]

correlations = (
    df[correlation_columns]
    .corr()["Trade_Return"]
    .sort_values(
        ascending=False
    )
)

print(
    correlations.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)

correlations.to_csv(
    OUTPUT_DIR / "signal_correlations.csv"
)


# ============================================================
# STABILITY TEST
# ============================================================

print_section(
    "SIGNAL STABILITY ACROSS PERIODS"
)

# Test whether high-conviction conditions work
# in at least 3 of the 5 periods.

stability_conditions = {

    "BuyProb>=0.60":
        df[buy_prob_col] >= 0.60,

    "BuyProb>=0.65":
        df[buy_prob_col] >= 0.65,

    "BuyProb>=0.70":
        df[buy_prob_col] >= 0.70,

    "Expected>=4":
        df[expected_col] >= 4,

    "Expected>=6":
        df[expected_col] >= 6,

    "Rank>=0.30":
        df[rank_col] >= 0.30,

    "Rank>=0.40":
        df[rank_col] >= 0.40,

    "Prob>=0.65 & Expected>=2":
        (
            (df[buy_prob_col] >= 0.65)
            &
            (df[expected_col] >= 2)
        ),

    "Prob>=0.70 & Expected>=4":
        (
            (df[buy_prob_col] >= 0.70)
            &
            (df[expected_col] >= 4)
        ),
}

stability_rows = []

for name, mask in stability_conditions.items():

    subset = df.loc[mask]

    if subset.empty:
        continue

    period_results = []

    for period, period_df in subset.groupby(
        period_col
    ):

        period_results.append(
            {
                "Period": period,
                "Trades": len(period_df),
                "Return":
                    period_df[
                        "Trade_Return"
                    ].mean(),
                "Win_Rate":
                    period_df["Win"].mean(),
            }
        )

    period_df = pd.DataFrame(
        period_results
    )

    profitable_periods = (
        period_df["Return"] > 0
    ).sum()

    stability_rows.append(
        {
            "Condition": name,
            "Total_Trades": len(subset),
            "Average_Return":
                subset["Trade_Return"].mean(),
            "Win_Rate":
                subset["Win"].mean(),
            "Profitable_Periods":
                profitable_periods,
            "Periods_Tested":
                len(period_df),
            "Stability_Rate":
                profitable_periods
                / len(period_df),
        }
    )

stability_df = pd.DataFrame(
    stability_rows
)

print(
    stability_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)

stability_df.to_csv(
    OUTPUT_DIR / "signal_stability.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print_section(
    "V9 SIGNAL ANALYSIS COMPLETE"
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFiles created:")
print("  period_signal_stats.csv")
print("  buy_probability_analysis.csv")
print("  expected_return_analysis.csv")
print("  technical_score_analysis.csv")
print("  rank_score_analysis.csv")
print("  stock_analysis.csv")
print("  regime_analysis.csv")
print("  period_regime_analysis.csv")
print("  high_conviction_tests.csv")
print("  combination_tests.csv")
print("  holding_period_analysis.csv")
print("  signal_correlations.csv")
print("  signal_stability.csv")

print("\n" + "=" * 70)