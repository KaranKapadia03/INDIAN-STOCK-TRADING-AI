from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

AI_EQUITY_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "strategy_v5_validation"
    / "validation_equity.csv"
)

NIFTY_FILE = BASE_DIR / "data" / "raw" / "nifty50.csv"

OUTPUT_DIR = BASE_DIR / "data" / "models" / "benchmark_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000

# Same approximate round-trip friction as the AI strategy:
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
ROUND_TRIP_COST = 2 * (TRANSACTION_COST + SLIPPAGE)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    columns = {str(c).lower().strip(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    for column in df.columns:
        column_lower = str(column).lower().strip()

        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


def calculate_metrics(equity):
    """Calculate portfolio performance metrics."""

    equity = equity.dropna().copy()

    if len(equity) < 2:
        return {
            "Final Capital": np.nan,
            "Return": np.nan,
            "CAGR": np.nan,
            "Max Drawdown": np.nan,
            "Sharpe": np.nan,
            "Volatility": np.nan,
        }

    start_value = equity.iloc[0]
    final_value = equity.iloc[-1]

    total_return = final_value / start_value - 1

    days = (equity.index[-1] - equity.index[0]).days

    if days > 0:
        cagr = (final_value / start_value) ** (365.25 / days) - 1
    else:
        cagr = np.nan

    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    volatility = daily_returns.std() * np.sqrt(252)

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Final Capital": final_value,
        "Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Volatility": volatility,
    }


# ============================================================
# LOAD AI VALIDATION EQUITY
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V6 - AI vs NIFTY 50 BENCHMARK")
print("=" * 70)

print("\nLoading AI validation equity...")

if not AI_EQUITY_FILE.exists():
    raise FileNotFoundError(
        f"AI validation equity not found:\n{AI_EQUITY_FILE}\n"
        "Run V5 validation first."
    )

ai_df = pd.read_csv(AI_EQUITY_FILE)

date_col = find_column(ai_df, ["Date", "Datetime", "Timestamp"])
equity_col = find_column(
    ai_df,
    ["Equity", "Portfolio_Value", "PortfolioValue", "Capital"]
)

if date_col is None:
    raise ValueError(
        f"Could not find date column in {AI_EQUITY_FILE}"
    )

if equity_col is None:
    raise ValueError(
        f"Could not find equity column in {AI_EQUITY_FILE}"
    )

ai_df[date_col] = pd.to_datetime(ai_df[date_col])

ai_df = ai_df.sort_values(date_col)

ai_df = ai_df.set_index(date_col)

ai_equity = ai_df[equity_col].astype(float)

validation_start = ai_equity.index.min()
validation_end = ai_equity.index.max()

print(f"AI validation period:")
print(f"{validation_start.date()} -> {validation_end.date()}")


# ============================================================
# LOAD NIFTY
# ============================================================

print("\nLoading NIFTY 50 data...")

if not NIFTY_FILE.exists():
    raise FileNotFoundError(
        f"NIFTY data not found:\n{NIFTY_FILE}"
    )

nifty_df = pd.read_csv(NIFTY_FILE)

nifty_date_col = find_column(
    nifty_df,
    ["Date", "Datetime", "Timestamp"]
)

nifty_close_col = find_column(
    nifty_df,
    ["Close", "close"]
)

if nifty_date_col is None:
    raise ValueError("Could not find NIFTY date column.")

if nifty_close_col is None:
    raise ValueError("Could not find NIFTY Close column.")

nifty_df[nifty_date_col] = pd.to_datetime(
    nifty_df[nifty_date_col]
)

nifty_df = nifty_df.sort_values(nifty_date_col)

nifty_df = nifty_df.set_index(nifty_date_col)

nifty_df = nifty_df.loc[
    (nifty_df.index >= validation_start)
    & (nifty_df.index <= validation_end)
]

nifty_df[nifty_close_col] = pd.to_numeric(
    nifty_df[nifty_close_col],
    errors="coerce"
)

nifty_df = nifty_df.dropna(subset=[nifty_close_col])

if nifty_df.empty:
    raise ValueError(
        "No NIFTY data overlaps the AI validation period."
    )

print(
    f"NIFTY data period:\n"
    f"{nifty_df.index.min().date()} -> "
    f"{nifty_df.index.max().date()}"
)


# ============================================================
# CREATE NIFTY BUY & HOLD
# ============================================================

nifty_start_price = nifty_df[nifty_close_col].iloc[0]

nifty_df["NIFTY_Equity"] = (
    INITIAL_CAPITAL
    * nifty_df[nifty_close_col]
    / nifty_start_price
)

# Cost-adjusted version.
# Apply the same approximate entry + exit friction.
nifty_df["NIFTY_Equity_Adjusted"] = (
    nifty_df["NIFTY_Equity"]
    * (1 - ROUND_TRIP_COST)
)


# ============================================================
# ALIGN AI EQUITY
# ============================================================

ai_equity = ai_equity.sort_index()

# Normalize AI equity to ₹100,000 at the beginning of validation.
ai_equity = (
    ai_equity
    / ai_equity.iloc[0]
    * INITIAL_CAPITAL
)

combined = pd.concat(
    [
        ai_equity.rename("AI_Equity"),
        nifty_df["NIFTY_Equity"],
        nifty_df["NIFTY_Equity_Adjusted"],
    ],
    axis=1
)

combined = combined.ffill().dropna()

combined.to_csv(
    OUTPUT_DIR / "ai_vs_nifty_equity.csv"
)


# ============================================================
# METRICS
# ============================================================

ai_metrics = calculate_metrics(combined["AI_Equity"])

nifty_metrics = calculate_metrics(
    combined["NIFTY_Equity"]
)

nifty_adjusted_metrics = calculate_metrics(
    combined["NIFTY_Equity_Adjusted"]
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "AI V5 Validation": ai_metrics,
        "NIFTY 50 Buy & Hold": nifty_metrics,
        "NIFTY 50 Cost Adjusted": nifty_adjusted_metrics,
    }
)

comparison = comparison.T

comparison["Outperformance_vs_NIFTY"] = (
    comparison["Return"]
    - nifty_metrics["Return"]
)

comparison.to_csv(
    OUTPUT_DIR / "benchmark_metrics.csv"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

print(
    f"\n{'Metric':<25}"
    f"{'AI V5':>15}"
    f"{'NIFTY 50':>15}"
    f"{'NIFTY Adj.':>15}"
)

print("-" * 70)

metrics_to_print = [
    ("Final Capital", "₹", ""),
    ("Return", "", "%"),
    ("CAGR", "", "%"),
    ("Max Drawdown", "", "%"),
    ("Sharpe", "", ""),
    ("Volatility", "", "%"),
]

for metric, prefix, suffix in metrics_to_print:

    ai_value = ai_metrics[metric]
    nifty_value = nifty_metrics[metric]
    nifty_adj_value = nifty_adjusted_metrics[metric]

    if metric == "Final Capital":
        print(
            f"{metric:<25}"
            f"{prefix}{ai_value:>13,.2f}"
            f"{prefix}{nifty_value:>13,.2f}"
            f"{prefix}{nifty_adj_value:>13,.2f}"
        )

    elif metric in ["Return", "CAGR", "Max Drawdown", "Volatility"]:
        print(
            f"{metric:<25}"
            f"{ai_value * 100:>13.2f}%"
            f"{nifty_value * 100:>13.2f}%"
            f"{nifty_adj_value * 100:>13.2f}%"
        )

    else:
        print(
            f"{metric:<25}"
            f"{ai_value:>14.2f}"
            f"{nifty_value:>14.2f}"
            f"{nifty_adj_value:>14.2f}"
        )


# ============================================================
# OUTPERFORMANCE
# ============================================================

return_difference = (
    ai_metrics["Return"]
    - nifty_metrics["Return"]
)

cagr_difference = (
    ai_metrics["CAGR"]
    - nifty_metrics["CAGR"]
)

drawdown_difference = (
    ai_metrics["Max Drawdown"]
    - nifty_metrics["Max Drawdown"]
)

print("\n" + "=" * 70)
print("AI ADVANTAGE vs NIFTY 50")
print("=" * 70)

print(
    f"\nReturn difference: "
    f"{return_difference * 100:+.2f}%"
)

print(
    f"CAGR difference: "
    f"{cagr_difference * 100:+.2f}%"
)

print(
    f"Max drawdown difference: "
    f"{drawdown_difference * 100:+.2f}%"
)

if return_difference > 0:
    print("\nRESULT: AI OUTPERFORMED NIFTY 50")
else:
    print("\nRESULT: AI UNDERPERFORMED NIFTY 50")


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Validation_Start": validation_start.date(),
            "Validation_End": validation_end.date(),
            "AI_Final_Capital": ai_metrics["Final Capital"],
            "AI_Return": ai_metrics["Return"],
            "AI_CAGR": ai_metrics["CAGR"],
            "AI_Max_Drawdown": ai_metrics["Max Drawdown"],
            "AI_Sharpe": ai_metrics["Sharpe"],
            "NIFTY_Final_Capital": nifty_metrics["Final Capital"],
            "NIFTY_Return": nifty_metrics["Return"],
            "NIFTY_CAGR": nifty_metrics["CAGR"],
            "NIFTY_Max_Drawdown": nifty_metrics["Max Drawdown"],
            "NIFTY_Sharpe": nifty_metrics["Sharpe"],
            "Return_Outperformance": return_difference,
            "CAGR_Outperformance": cagr_difference,
        }
    ]
)

summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)


print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("V6 BENCHMARK COMPARISON COMPLETE")
print("=" * 70)