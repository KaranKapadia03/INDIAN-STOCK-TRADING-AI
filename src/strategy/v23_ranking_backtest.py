from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_FILE = Path(
    "data/models/v22_excess_return/v22_predictions.csv"
)

MARKET_FILE = Path(
    "data/raw/nifty50.csv"
)

OUTPUT_DIR = Path(
    "data/models/v23_ranking_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_CAPITAL = 100_000.0

TOP_N = 5

HOLDING_DAYS = 20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

ONE_WAY_COST = (
    TRANSACTION_COST
    + SLIPPAGE
)


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    try:
        if df["Date"].dt.tz is not None:
            df["Date"] = (
                df["Date"]
                .dt.tz_localize(None)
            )
    except Exception:
        pass

    df["Date"] = (
        df["Date"]
        .dt.normalize()
    )

    return df


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    df = normalize_dates(df)

    df = df[
        df["Model"] == "RandomForest"
    ].copy()

    df = (
        df
        .sort_values(
            ["Date", "Symbol"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Random Forest predictions: "
        f"{len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"-> "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# LOAD NIFTY
# ============================================================

def load_nifty():

    df = pd.read_csv(
        MARKET_FILE
    )

    df = normalize_dates(df)

    close_column = None

    for column in df.columns:

        if str(column).lower() == "close":

            close_column = column
            break

    if close_column is None:

        raise ValueError(
            "Could not find Close column "
            "in nifty50.csv"
        )

    df = (
        df[
            [
                "Date",
                close_column
            ]
        ]
        .rename(
            columns={
                close_column:
                "NIFTY_Close"
            }
        )
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    df[
        "NIFTY_Daily_Return"
    ] = (
        df["NIFTY_Close"]
        .pct_change()
    )

    return df


# ============================================================
# CREATE RANKED PORTFOLIO
# ============================================================

def create_portfolio(
    predictions
):

    rows = []

    dates = sorted(
        predictions["Date"].unique()
    )

    # Rebalance every 20 trading days.

    rebalance_dates = dates[
        ::HOLDING_DAYS
    ]

    for rebalance_date in (
        rebalance_dates
    ):

        day = predictions[
            predictions["Date"]
            == rebalance_date
        ].copy()

        if len(day) < TOP_N:
            continue

        day = (
            day
            .sort_values(
                "Predicted_Excess_Return",
                ascending=False
            )
            .head(TOP_N)
        )

        for rank, (_, row) in enumerate(
            day.iterrows(),
            start=1
        ):

            rows.append({

                "Entry_Date":
                    rebalance_date,

                "Symbol":
                    row["Symbol"],

                "Rank":
                    rank,

                "Predicted_Excess_Return":
                    row[
                        "Predicted_Excess_Return"
                    ],

                "Actual_Excess_Return":
                    row[
                        "Excess_Return_20D"
                    ],

                "Stock_Return_20D":
                    row[
                        "Stock_Return_20D"
                    ],
            })

    return pd.DataFrame(rows)


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    portfolio
):

    portfolio = portfolio.copy()

    # --------------------------------------------------------
    # Convert percentage to decimal.
    #
    # Example:
    # +5% -> 0.05
    # -3% -> -0.03
    # --------------------------------------------------------

    portfolio[
        "Gross_Return"
    ] = (
        portfolio[
            "Stock_Return_20D"
        ] / 100.0
    )

    # --------------------------------------------------------
    # Apply entry + exit costs.
    #
    # Wealth after trade:
    #
    # (1 + return)
    # × (1 - entry cost)
    # × (1 - exit cost)
    #
    # Then subtract 1 to get the actual return.
    # --------------------------------------------------------

    portfolio[
        "Net_Return"
    ] = (

        (
            1
            + portfolio[
                "Gross_Return"
            ]
        )

        * (1 - ONE_WAY_COST)

        * (1 - ONE_WAY_COST)

        - 1
    )

    portfolio[
        "Net_Return_Pct"
    ] = (
        portfolio[
            "Net_Return"
        ] * 100
    )

    # --------------------------------------------------------
    # Equal-weight portfolio return
    # --------------------------------------------------------

    portfolio_returns = (
        portfolio
        .groupby("Entry_Date")[
            "Net_Return"
        ]
        .mean()
        .sort_index()
    )

    # --------------------------------------------------------
    # Compound portfolio
    # --------------------------------------------------------

    equity = INITIAL_CAPITAL

    equity_rows = []

    for date, ret in (
        portfolio_returns.items()
    ):

        equity *= (
            1 + ret
        )

        equity_rows.append({

            "Date":
                date,

            "Portfolio_Return":
                ret,

            "Portfolio_Return_Pct":
                ret * 100,

            "Equity":
                equity,
        })

    equity_curve = pd.DataFrame(
        equity_rows
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_equity = (
        equity_curve[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    days = (
        equity_curve["Date"].iloc[-1]
        - equity_curve["Date"].iloc[0]
    ).days

    years = (
        days / 365.25
    )

    if years > 0:

        cagr = (
            (
                final_equity
                / INITIAL_CAPITAL
            )
            ** (1 / years)
            - 1
        )

    else:

        cagr = np.nan

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        equity_curve[
            "Equity"
        ].cummax()
    )

    drawdown = (
        equity_curve[
            "Equity"
        ]
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    if (
        len(portfolio_returns) > 1
        and portfolio_returns.std() > 0
    ):

        sharpe = (
            portfolio_returns.mean()
            / portfolio_returns.std()
            * np.sqrt(
                252 / HOLDING_DAYS
            )
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Trade statistics
    # --------------------------------------------------------

    win_rate = (
        portfolio[
            "Net_Return"
        ] > 0
    ).mean()

    gross_profit = (
        portfolio.loc[
            portfolio["Net_Return"] > 0,
            "Net_Return"
        ].sum()
    )

    gross_loss = abs(
        portfolio.loc[
            portfolio["Net_Return"] < 0,
            "Net_Return"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    average_trade = (
        portfolio[
            "Net_Return"
        ].mean()
    )

    median_trade = (
        portfolio[
            "Net_Return"
        ].median()
    )

    metrics = {

        "Initial_Capital":
            INITIAL_CAPITAL,

        "Final_Equity":
            final_equity,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Win_Rate":
            win_rate,

        "Profit_Factor":
            profit_factor,

        "Average_Trade":
            average_trade,

        "Median_Trade":
            median_trade,

        "Trades":
            len(portfolio),

        "Rebalance_Period":
            HOLDING_DAYS,

        "Top_N":
            TOP_N,
    }

    return (
        portfolio,
        equity_curve,
        metrics
    )


# ============================================================
# NIFTY BUY & HOLD
# ============================================================

def calculate_nifty_return(
    nifty,
    start_date,
    end_date
):

    nifty = nifty.copy()

    nifty = nifty[
        (
            nifty["Date"]
            >= start_date
        )
        &
        (
            nifty["Date"]
            <= end_date
        )
    ].copy()

    if len(nifty) < 2:

        return np.nan

    start_price = (
        nifty[
            "NIFTY_Close"
        ].iloc[0]
    )

    end_price = (
        nifty[
            "NIFTY_Close"
        ].iloc[-1]
    )

    return (
        end_price
        / start_price
        - 1
    )


# ============================================================
# STOCK ATTRIBUTION
# ============================================================

def stock_attribution(
    portfolio
):

    result = (
        portfolio
        .groupby("Symbol")
        .agg(

            Trades=(
                "Net_Return",
                "count"
            ),

            Average_Return=(
                "Net_Return",
                "mean"
            ),

            Total_Return=(
                "Net_Return",
                "sum"
            ),

            Win_Rate=(
                "Net_Return",
                lambda x:
                (x > 0).mean()
            )
        )
        .sort_values(
            "Total_Return",
            ascending=False
        )
    )

    return result


# ============================================================
# TOP-N SENSITIVITY
# ============================================================

def top_n_analysis(
    predictions
):

    results = []

    dates = sorted(
        predictions["Date"].unique()
    )

    rebalance_dates = dates[
        ::HOLDING_DAYS
    ]

    for top_n in [3, 5, 7, 10]:

        returns = []

        for date in rebalance_dates:

            day = predictions[
                predictions["Date"] == date
            ].copy()

            if len(day) < top_n:
                continue

            selected = (
                day
                .sort_values(
                    "Predicted_Excess_Return",
                    ascending=False
                )
                .head(top_n)
            )

            gross = (
                selected[
                    "Stock_Return_20D"
                ] / 100
            )

            net = (
                (
                    1 + gross
                )
                * (1 - ONE_WAY_COST)
                * (1 - ONE_WAY_COST)
                - 1
            )

            returns.append(
                net.mean()
            )

        if returns:

            equity = np.prod(
                1 + np.array(returns)
            )

            total_return = (
                equity - 1
            )

            results.append({

                "Top_N":
                    top_n,

                "Rebalances":
                    len(returns),

                "Final_Multiple":
                    equity,

                "Total_Return":
                    total_return,

                "Average_Rebalance_Return":
                    np.mean(returns),

                "Win_Rate":
                    np.mean(
                        np.array(returns) > 0
                    ),
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metrics,
    nifty_return
):

    print("\n")
    print("=" * 70)
    print("V23 RANKING PORTFOLIO BACKTEST")
    print("=" * 70)

    print(
        f"Initial Capital: "
        f"₹{metrics['Initial_Capital']:,.2f}"
    )

    print(
        f"Final Equity:    "
        f"₹{metrics['Final_Equity']:,.2f}"
    )

    print(
        f"Total Return:    "
        f"{metrics['Total_Return'] * 100:.2f}%"
    )

    print(
        f"CAGR:             "
        f"{metrics['CAGR'] * 100:.2f}%"
    )

    print(
        f"Max Drawdown:     "
        f"{metrics['Max_Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe:           "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Win Rate:         "
        f"{metrics['Win_Rate'] * 100:.2f}%"
    )

    print(
        f"Profit Factor:    "
        f"{metrics['Profit_Factor']:.2f}"
    )

    print(
        f"Average Trade:    "
        f"{metrics['Average_Trade'] * 100:.3f}%"
    )

    print(
        f"Median Trade:     "
        f"{metrics['Median_Trade'] * 100:.3f}%"
    )

    print(
        f"Trades:           "
        f"{metrics['Trades']}"
    )

    print(
        f"NIFTY Return:     "
        f"{nifty_return * 100:.2f}%"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("V23 - RANKING PORTFOLIO BACKTEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    predictions = (
        load_predictions()
    )

    nifty = (
        load_nifty()
    )

    # --------------------------------------------------------
    # Portfolio
    # --------------------------------------------------------

    portfolio = (
        create_portfolio(
            predictions
        )
    )

    print(
        f"\nPortfolio positions: "
        f"{len(portfolio):,}"
    )

    print(
        f"Top N stocks: "
        f"{TOP_N}"
    )

    print(
        f"Holding period: "
        f"{HOLDING_DAYS} trading days"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.2f}%"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    (
        portfolio,
        equity_curve,
        metrics
    ) = run_backtest(
        portfolio
    )

    # --------------------------------------------------------
    # NIFTY
    # --------------------------------------------------------

    nifty_return = (
        calculate_nifty_return(
            nifty,
            equity_curve["Date"].iloc[0],
            equity_curve["Date"].iloc[-1]
        )
    )

    metrics[
        "NIFTY_Return"
    ] = nifty_return

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        metrics,
        nifty_return
    )

    # --------------------------------------------------------
    # Attribution
    # --------------------------------------------------------

    attribution = (
        stock_attribution(
            portfolio
        )
    )

    print("\n")
    print("=" * 70)
    print("STOCK ATTRIBUTION")
    print("=" * 70)

    print(
        attribution.to_string()
    )

    # --------------------------------------------------------
    # Top-N sensitivity
    # --------------------------------------------------------

    sensitivity = (
        top_n_analysis(
            predictions
        )
    )

    print("\n")
    print("=" * 70)
    print("TOP-N SENSITIVITY")
    print("=" * 70)

    print(
        sensitivity.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    portfolio_file = (
        OUTPUT_DIR
        / "v23_trade_log.csv"
    )

    equity_file = (
        OUTPUT_DIR
        / "v23_equity_curve.csv"
    )

    attribution_file = (
        OUTPUT_DIR
        / "v23_stock_attribution.csv"
    )

    sensitivity_file = (
        OUTPUT_DIR
        / "v23_top_n_sensitivity.csv"
    )

    metrics_file = (
        OUTPUT_DIR
        / "v23_metrics.csv"
    )

    portfolio.to_csv(
        portfolio_file,
        index=False
    )

    equity_curve.to_csv(
        equity_file,
        index=False
    )

    attribution.to_csv(
        attribution_file
    )

    sensitivity.to_csv(
        sensitivity_file,
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_file,
        index=False
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        portfolio_file.resolve()
    )

    print(
        equity_file.resolve()
    )

    print(
        attribution_file.resolve()
    )

    print(
        sensitivity_file.resolve()
    )

    print(
        metrics_file.resolve()
    )

    print("\n")
    print("=" * 70)
    print("V23 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()