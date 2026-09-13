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
    / "strategy_v5_validation"
    / "validation_trades.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "strategy_v7_attribution"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000


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
# LOAD TRADES
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V7 - TRADE ATTRIBUTION & ROBUSTNESS ANALYSIS")
print("=" * 70)

print("\nLoading validation trades...")

if not TRADE_FILE.exists():
    raise FileNotFoundError(
        f"\nValidation trades not found:\n{TRADE_FILE}\n"
        "Run V5 validation first."
    )

df = pd.read_csv(TRADE_FILE)

print(f"Trades loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

symbol_col = find_column(
    df,
    ["Symbol", "Ticker", "Stock"]
)

return_col = find_column(
    df,
    [
        "Return",
        "Trade_Return",
        "TradeReturn",
        "PnL_Percent",
        "Return_Percent",
    ]
)

pnl_col = find_column(
    df,
    [
        "PnL",
        "P&L",
        "Profit",
        "Profit_Loss",
        "Net_PnL",
    ]
)

regime_col = find_column(
    df,
    [
        "Market_Regime",
        "Regime",
    ]
)

exit_col = find_column(
    df,
    [
        "Exit_Reason",
        "ExitReason",
        "Reason",
    ]
)

entry_date_col = find_column(
    df,
    [
        "Entry_Date",
        "EntryDate",
        "Entry",
    ]
)

exit_date_col = find_column(
    df,
    [
        "Exit_Date",
        "ExitDate",
        "Exit",
    ]
)

if symbol_col is None:
    raise ValueError("Could not identify stock/symbol column.")

if return_col is None:
    raise ValueError("Could not identify trade return column.")

if pnl_col is None:
    raise ValueError("Could not identify PnL column.")

if regime_col is None:
    print("\nWARNING: Regime column not found.")

if exit_col is None:
    print("\nWARNING: Exit reason column not found.")


# ============================================================
# CLEAN DATA
# ============================================================

df[symbol_col] = df[symbol_col].astype(str)

df[return_col] = pd.to_numeric(
    df[return_col],
    errors="coerce"
)

df[pnl_col] = pd.to_numeric(
    df[pnl_col],
    errors="coerce"
)

df = df.dropna(
    subset=[return_col, pnl_col]
).copy()

# Convert returns to decimal if stored as percentages.
#
# Example:
# 5.2 means +5.2%
# 0.052 means +5.2%
#
# Detect automatically.

if df[return_col].abs().median() > 1:
    df["Return_Decimal"] = df[return_col] / 100
else:
    df["Return_Decimal"] = df[return_col]

df["Win"] = df["Return_Decimal"] > 0


# ============================================================
# BASIC STATISTICS
# ============================================================

print_section("OVERALL TRADE STATISTICS")

total_trades = len(df)

winning_trades = int(df["Win"].sum())
losing_trades = total_trades - winning_trades

win_rate = winning_trades / total_trades

avg_return = df["Return_Decimal"].mean()

median_return = df["Return_Decimal"].median()

best_trade = df["Return_Decimal"].max()

worst_trade = df["Return_Decimal"].min()

total_pnl = df[pnl_col].sum()

print(f"\nTotal trades:        {total_trades}")
print(f"Winning trades:      {winning_trades}")
print(f"Losing trades:       {losing_trades}")
print(f"Win rate:            {win_rate:.2%}")
print(f"Average trade:       {avg_return:.2%}")
print(f"Median trade:        {median_return:.2%}")
print(f"Best trade:          {best_trade:.2%}")
print(f"Worst trade:         {worst_trade:.2%}")
print(f"Total PnL:            ₹{total_pnl:,.2f}")


# ============================================================
# PROFIT / LOSS CONCENTRATION
# ============================================================

print_section("PROFIT CONCENTRATION")

positive_pnl = df.loc[
    df[pnl_col] > 0,
    pnl_col
].sum()

negative_pnl = df.loc[
    df[pnl_col] < 0,
    pnl_col
].sum()

gross_profit = positive_pnl

gross_loss = abs(negative_pnl)

if gross_loss > 0:
    profit_factor = gross_profit / gross_loss
else:
    profit_factor = np.inf

print(f"\nGross profit:        ₹{gross_profit:,.2f}")
print(f"Gross loss:          ₹{gross_loss:,.2f}")
print(f"Profit factor:       {profit_factor:.2f}")


# ============================================================
# TOP TRADE CONTRIBUTION
# ============================================================

sorted_trades = df.sort_values(
    pnl_col,
    ascending=False
).reset_index(drop=True)

total_positive_pnl = max(
    positive_pnl,
    1e-9
)

top_1_profit = sorted_trades.iloc[:1][pnl_col].sum()
top_3_profit = sorted_trades.iloc[:3][pnl_col].sum()
top_5_profit = sorted_trades.iloc[:5][pnl_col].sum()
top_10_profit = sorted_trades.iloc[:10][pnl_col].sum()

print("\nContribution from winning trades:")
print(
    f"Top 1 trade:         "
    f"{top_1_profit / total_positive_pnl:.2%}"
)
print(
    f"Top 3 trades:        "
    f"{top_3_profit / total_positive_pnl:.2%}"
)
print(
    f"Top 5 trades:        "
    f"{top_5_profit / total_positive_pnl:.2%}"
)
print(
    f"Top 10 trades:       "
    f"{top_10_profit / total_positive_pnl:.2%}"
)


# ============================================================
# STOCK ATTRIBUTION
# ============================================================

print_section("PERFORMANCE BY STOCK")

stock_stats = (
    df.groupby(symbol_col)
    .agg(
        Trades=(return_col, "count"),
        Total_PnL=(pnl_col, "sum"),
        Average_Return=("Return_Decimal", "mean"),
        Median_Return=("Return_Decimal", "median"),
        Win_Rate=("Win", "mean"),
        Best_Trade=("Return_Decimal", "max"),
        Worst_Trade=("Return_Decimal", "min"),
    )
    .sort_values(
        "Total_PnL",
        ascending=False
    )
)

print(
    stock_stats.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

stock_stats.to_csv(
    OUTPUT_DIR / "stock_attribution.csv"
)


# ============================================================
# REGIME ATTRIBUTION
# ============================================================

if regime_col is not None:

    print_section("PERFORMANCE BY MARKET REGIME")

    regime_stats = (
        df.groupby(regime_col)
        .agg(
            Trades=(return_col, "count"),
            Total_PnL=(pnl_col, "sum"),
            Average_Return=("Return_Decimal", "mean"),
            Median_Return=("Return_Decimal", "median"),
            Win_Rate=("Win", "mean"),
            Best_Trade=("Return_Decimal", "max"),
            Worst_Trade=("Return_Decimal", "min"),
        )
        .sort_values(
            "Total_PnL",
            ascending=False
        )
    )

    print(
        regime_stats.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    regime_stats.to_csv(
        OUTPUT_DIR / "regime_attribution.csv"
    )


# ============================================================
# EXIT REASON
# ============================================================

if exit_col is not None:

    print_section("EXIT REASON ANALYSIS")

    exit_stats = (
        df.groupby(exit_col)
        .agg(
            Trades=(return_col, "count"),
            Total_PnL=(pnl_col, "sum"),
            Average_Return=("Return_Decimal", "mean"),
            Win_Rate=("Win", "mean"),
        )
        .sort_values(
            "Total_PnL",
            ascending=False
        )
    )

    print(
        exit_stats.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    exit_stats.to_csv(
        OUTPUT_DIR / "exit_reason_analysis.csv"
    )


# ============================================================
# RETURN DISTRIBUTION
# ============================================================

print_section("RETURN DISTRIBUTION")

thresholds = [
    -0.10,
    -0.05,
    -0.02,
    0,
    0.02,
    0.05,
    0.10,
]

for threshold in thresholds:

    if threshold < 0:
        count = (
            df["Return_Decimal"] <= threshold
        ).sum()

        print(
            f"Trades <= {threshold:.0%}: "
            f"{count}"
        )

    else:
        count = (
            df["Return_Decimal"] >= threshold
        ).sum()

        print(
            f"Trades >= {threshold:.0%}: "
            f"{count}"
        )


# ============================================================
# REMOVE TOP WINNERS ROBUSTNESS TEST
# ============================================================

print_section("ROBUSTNESS - REMOVE TOP WINNERS")

def calculate_total_return_from_trades(trades):
    """
    Approximate compounded return assuming each trade
    compounds the portfolio.
    """

    if trades.empty:
        return 0.0

    equity = 1.0

    for trade_return in trades["Return_Decimal"]:
        equity *= (1 + trade_return)

    return equity - 1


baseline_return = calculate_total_return_from_trades(df)

robustness_rows = []

for remove_count in [1, 2, 3, 5, 10]:

    remaining = sorted_trades.iloc[
        remove_count:
    ].copy()

    result = calculate_total_return_from_trades(
        remaining
    )

    robustness_rows.append(
        {
            "Removed_Top_Trades": remove_count,
            "Remaining_Trades": len(remaining),
            "Compounded_Return": result,
            "Change_vs_Baseline": result - baseline_return,
        }
    )

robustness = pd.DataFrame(
    robustness_rows
)

print(
    robustness.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

robustness.to_csv(
    OUTPUT_DIR / "winner_removal_robustness.csv",
    index=False
)


# ============================================================
# REMOVE WORST LOSERS ROBUSTNESS TEST
# ============================================================

print_section("ROBUSTNESS - REMOVE WORST LOSERS")

worst_sorted = df.sort_values(
    "Return_Decimal",
    ascending=True
).reset_index(drop=True)

loss_robustness = []

for remove_count in [1, 2, 3, 5]:

    remaining = worst_sorted.iloc[
        remove_count:
    ].copy()

    result = calculate_total_return_from_trades(
        remaining
    )

    loss_robustness.append(
        {
            "Removed_Worst_Trades": remove_count,
            "Remaining_Trades": len(remaining),
            "Compounded_Return": result,
            "Change_vs_Baseline": result - baseline_return,
        }
    )

loss_robustness = pd.DataFrame(
    loss_robustness
)

print(
    loss_robustness.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

loss_robustness.to_csv(
    OUTPUT_DIR / "loser_removal_robustness.csv",
    index=False
)


# ============================================================
# BEST / WORST TRADES
# ============================================================

print_section("TOP 10 WINNING TRADES")

display_columns = [
    c
    for c in [
        symbol_col,
        entry_date_col,
        exit_date_col,
        regime_col,
        return_col,
        pnl_col,
        exit_col,
    ]
    if c is not None
]

print(
    df.sort_values(
        "Return_Decimal",
        ascending=False
    )
    .head(10)[display_columns]
    .to_string(index=False)
)


print_section("TOP 10 LOSING TRADES")

print(
    df.sort_values(
        "Return_Decimal",
        ascending=True
    )
    .head(10)[display_columns]
    .to_string(index=False)
)


# ============================================================
# SAVE FULL CLEAN DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "clean_validation_trades.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "Total_Trades": total_trades,
    "Winning_Trades": winning_trades,
    "Losing_Trades": losing_trades,
    "Win_Rate": win_rate,
    "Average_Trade_Return": avg_return,
    "Median_Trade_Return": median_return,
    "Best_Trade": best_trade,
    "Worst_Trade": worst_trade,
    "Gross_Profit": gross_profit,
    "Gross_Loss": gross_loss,
    "Profit_Factor": profit_factor,
    "Top_1_Profit_Contribution": top_1_profit / total_positive_pnl,
    "Top_3_Profit_Contribution": top_3_profit / total_positive_pnl,
    "Top_5_Profit_Contribution": top_5_profit / total_positive_pnl,
    "Top_10_Profit_Contribution": top_10_profit / total_positive_pnl,
}

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print_section("V7 COMPLETE")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFiles created:")
print("  stock_attribution.csv")
print("  regime_attribution.csv")
print("  exit_reason_analysis.csv")
print("  winner_removal_robustness.csv")
print("  loser_removal_robustness.csv")
print("  clean_validation_trades.csv")
print("  summary.csv")

print("\n" + "=" * 70)