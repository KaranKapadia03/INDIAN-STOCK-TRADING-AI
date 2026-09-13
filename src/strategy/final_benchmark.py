
import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

AI_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "models",
    "execution_backtest",
    "equity.csv"
)

NIFTY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "nifty50.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models",
    "benchmark"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD AI EQUITY
# ============================================================

def load_ai():

    df = pd.read_csv(
        AI_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        errors="coerce"
    )

    df["date"] = (
        df["date"]
        .dt.tz_localize(None)
    )

    df = df.dropna(
        subset=["date"]
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD NIFTY
# ============================================================

def load_nifty():

    df = pd.read_csv(
        NIFTY_FILE
    )

    # Find date column
    date_column = None

    for column in df.columns:

        if column.lower() in [
            "date",
            "datetime",
            "timestamp"
        ]:

            date_column = column
            break

    if date_column is None:

        raise ValueError(
            "No date column found in nifty50.csv"
        )

    # Find close column
    close_column = None

    for column in df.columns:

        name = (
            column
            .lower()
            .replace("_", " ")
            .strip()
        )

        if name in [
            "close",
            "adj close"
        ]:

            close_column = column
            break

    if close_column is None:

        raise ValueError(
            "No close column found in nifty50.csv"
        )

    df["date"] = pd.to_datetime(
        df[date_column],
        utc=True,
        errors="coerce"
    )

    df["date"] = (
        df["date"]
        .dt.tz_localize(None)
    )

    df["nifty_close"] = pd.to_numeric(
        df[close_column],
        errors="coerce"
    )

    df = df[
        [
            "date",
            "nifty_close"
        ]
    ]

    df = df.dropna()

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(series):

    peak = series.cummax()

    drawdown = (
        series
        /
        peak
        - 1
    )

    return drawdown.min() * 100


# ============================================================
# SHARPE
# ============================================================

def sharpe(series):

    returns = (
        series
        .pct_change()
        .dropna()
    )

    if (
        len(returns) < 2
        or
        returns.std() == 0
    ):

        return 0.0

    return (
        returns.mean()
        /
        returns.std()
    ) * np.sqrt(252)


# ============================================================
# CAGR
# ============================================================

def cagr(
    initial,
    final,
    start,
    end
):

    days = (
        end - start
    ).days

    years = (
        days / 365.25
    )

    if years <= 0:

        return 0.0

    return (
        (
            final / initial
        )
        **
        (1 / years)
        - 1
    ) * 100


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("INDIAN STOCK TRADING AI")
    print("AI vs NIFTY 50 BENCHMARK")
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ai = load_ai()

    nifty = load_nifty()

    print(
        f"\nAI rows    : {len(ai)}"
    )

    print(
        f"NIFTY rows : {len(nifty)}"
    )

    # --------------------------------------------------------
    # Common period
    # --------------------------------------------------------

    start = max(
        ai["date"].min(),
        nifty["date"].min()
    )

    end = min(
        ai["date"].max(),
        nifty["date"].max()
    )

    print(
        f"\nPeriod: "
        f"{start.date()} "
        f"to "
        f"{end.date()}"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    ai = ai[
        (
            ai["date"] >= start
        )
        &
        (
            ai["date"] <= end
        )
    ].copy()

    nifty = nifty[
        (
            nifty["date"] >= start
        )
        &
        (
            nifty["date"] <= end
        )
    ].copy()

    # --------------------------------------------------------
    # Normalize both to ₹100,000
    # --------------------------------------------------------

    ai_start = float(
        ai["portfolio_value"].iloc[0]
    )

    nifty_start = float(
        nifty["nifty_close"].iloc[0]
    )

    ai["AI_Value"] = (
        ai["portfolio_value"]
        /
        ai_start
        *
        100000
    )

    nifty["NIFTY_Value"] = (
        nifty["nifty_close"]
        /
        nifty_start
        *
        100000
    )

    # --------------------------------------------------------
    # Align dates
    # --------------------------------------------------------

    comparison = pd.merge_asof(

        ai[
            [
                "date",
                "AI_Value"
            ]
        ].sort_values("date"),

        nifty[
            [
                "date",
                "NIFTY_Value"
            ]
        ].sort_values("date"),

        on="date",

        direction="backward"
    )

    comparison = comparison.dropna()

    if comparison.empty:

        raise RuntimeError(
            "No overlapping dates found."
        )

    # --------------------------------------------------------
    # Values
    # --------------------------------------------------------

    ai_initial = float(
        comparison["AI_Value"].iloc[0]
    )

    ai_final = float(
        comparison["AI_Value"].iloc[-1]
    )

    nifty_initial = float(
        comparison["NIFTY_Value"].iloc[0]
    )

    nifty_final = float(
        comparison["NIFTY_Value"].iloc[-1]
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    ai_return = (
        ai_final
        /
        ai_initial
        - 1
    ) * 100

    nifty_return = (
        nifty_final
        /
        nifty_initial
        - 1
    ) * 100

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    period_start = (
        comparison["date"].iloc[0]
    )

    period_end = (
        comparison["date"].iloc[-1]
    )

    ai_cagr = cagr(
        ai_initial,
        ai_final,
        period_start,
        period_end
    )

    nifty_cagr = cagr(
        nifty_initial,
        nifty_final,
        period_start,
        period_end
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    ai_dd = max_drawdown(
        comparison["AI_Value"]
    )

    nifty_dd = max_drawdown(
        comparison["NIFTY_Value"]
    )

    ai_sharpe = sharpe(
        comparison["AI_Value"]
    )

    nifty_sharpe = sharpe(
        comparison["NIFTY_Value"]
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    return_difference = (
        ai_return
        -
        nifty_return
    )

    cagr_difference = (
        ai_cagr
        -
        nifty_cagr
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)

    print(
        f"{'Metric':<25}"
        f"{'AI Strategy':>20}"
        f"{'NIFTY 50':>20}"
    )

    print("-" * 70)

    print(
        f"{'Initial Capital':<25}"
        f"₹{ai_initial:>18,.2f}"
        f"₹{nifty_initial:>18,.2f}"
    )

    print(
        f"{'Final Value':<25}"
        f"₹{ai_final:>18,.2f}"
        f"₹{nifty_final:>18,.2f}"
    )

    print(
        f"{'Total Return':<25}"
        f"{ai_return:>19.2f}%"
        f"{nifty_return:>19.2f}%"
    )

    print(
        f"{'CAGR':<25}"
        f"{ai_cagr:>19.2f}%"
        f"{nifty_cagr:>19.2f}%"
    )

    print(
        f"{'Max Drawdown':<25}"
        f"{ai_dd:>19.2f}%"
        f"{nifty_dd:>19.2f}%"
    )

    print(
        f"{'Sharpe Ratio':<25}"
        f"{ai_sharpe:>20.2f}"
        f"{nifty_sharpe:>20.2f}"
    )

    # ========================================================
    # OUTPERFORMANCE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("AI vs NIFTY")
    print("=" * 100)

    print(
        f"Return difference : "
        f"{return_difference:+.2f}%"
    )

    print(
        f"CAGR difference   : "
        f"{cagr_difference:+.2f}%"
    )

    if return_difference > 0:

        print(
            "\nVERDICT: "
            "AI beat NIFTY 50 on total return."
        )

    elif return_difference < 0:

        print(
            "\nVERDICT: "
            "NIFTY 50 beat the AI strategy."
        )

    else:

        print(
            "\nVERDICT: "
            "AI and NIFTY had equal returns."
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        "ai_vs_nifty.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        output_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

