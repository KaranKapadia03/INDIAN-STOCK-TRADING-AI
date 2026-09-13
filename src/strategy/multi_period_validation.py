from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "regime_walk_forward"
    / "regime_walk_forward_predictions.csv"
)

MARKET_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "strategy_v8_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000

# Number of sequential validation periods.
N_PERIODS = 5

# Maximum portfolio positions.
MAX_POSITIONS = 4

# Maximum total capital invested.
MAX_EXPOSURE = 0.80

# Risk management.
STOP_LOSS = 0.06
TARGET = 0.12
MAX_HOLD_DAYS = 20

# Same trading friction as V5.
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005


# ============================================================
# V4 / V5 STRATEGY RULES
# ============================================================

def generate_signal(row):
    """
    Recreate the exact V4/V5 signal logic.
    """

    regime = str(row["Market_Regime"]).upper()

    buy_prob = float(row["Buy_Probability"])
    expected = float(row["Expected_Return_20D"])
    technical = float(row["Technical_Score"])

    # --------------------------------------------------------
    # Regime-specific entry requirements
    # --------------------------------------------------------

    if regime == "BEARISH":

        if (
            buy_prob >= 0.55
            and technical >= 0
            and expected >= 0
        ):
            return True

    elif regime == "NEUTRAL":

        if (
            buy_prob >= 0.65
            and technical >= 1
            and expected >= 2
        ):
            return True

    elif regime == "BULLISH":

        if (
            buy_prob >= 0.75
            and technical >= 2
            and expected >= 4
        ):
            return True

    return False


def calculate_rank(row):
    """
    Same ranking concept used by V4/V5.
    """

    buy_prob = float(row["Buy_Probability"])
    sell_prob = float(row["Sell_Probability"])
    expected = float(row["Expected_Return_20D"])
    technical = float(row["Technical_Score"])

    regime = str(row["Market_Regime"]).upper()

    regime_adjustment = {
        "BEARISH": 0.15,
        "NEUTRAL": 0.00,
        "BULLISH": -0.10,
    }.get(regime, 0.0)

    score = (
        0.40 * (buy_prob - sell_prob)
        + 0.30 * (expected / 10)
        + 0.20 * (technical / 4)
        + 0.10 * regime_adjustment
    )

    return score


# ============================================================
# MARKET DATA
# ============================================================

def load_market_data(symbols):

    market_data = {}

    for symbol in symbols:

        filename = (
            symbol
            .replace(".NS", "_NS")
            + ".csv"
        )

        path = MARKET_DIR / filename

        if not path.exists():
            continue

        df = pd.read_csv(path)

        date_col = None

        for col in df.columns:

            if str(col).lower() in [
                "date",
                "datetime",
                "timestamp",
            ]:
                date_col = col
                break

        if date_col is None:
            continue

        df[date_col] = pd.to_datetime(
            df[date_col]
        )

        df = df.sort_values(date_col)

        df = df.set_index(date_col)

        close_col = None

        for col in df.columns:

            if str(col).lower() == "close":
                close_col = col
                break

        if close_col is None:
            continue

        df["Close_Price"] = pd.to_numeric(
            df[close_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Close_Price"]
        )

        market_data[symbol] = df

    return market_data


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(predictions, market_data):

    dates = sorted(
        predictions["Date"].dropna().unique()
    )

    capital = INITIAL_CAPITAL

    open_positions = []

    trades = []

    equity_curve = []

    for current_date in dates:

        # ====================================================
        # UPDATE OPEN POSITIONS
        # ====================================================

        remaining_positions = []

        for position in open_positions:

            symbol = position["Symbol"]

            if symbol not in market_data:
                remaining_positions.append(position)
                continue

            market = market_data[symbol]

            available = market.loc[
                market.index >= current_date
            ]

            if available.empty:
                remaining_positions.append(position)
                continue

            current_price = float(
                available["Close_Price"].iloc[0]
            )

            entry_price = position["Entry_Price"]

            days_held = (
                pd.Timestamp(current_date)
                - pd.Timestamp(position["Entry_Date"])
            ).days

            current_return = (
                current_price / entry_price
                - 1
            )

            exit_reason = None

            # Target
            if current_return >= TARGET:
                exit_reason = "TARGET"

            # Stop
            elif current_return <= -STOP_LOSS:
                exit_reason = "STOP"

            # Time exit
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = "TIME"

            if exit_reason is not None:

                # Apply exit transaction cost + slippage.
                exit_value = (
                    position["Capital_Used"]
                    * (1 + current_return)
                    * (
                        1
                        - TRANSACTION_COST
                        - SLIPPAGE
                    )
                )

                pnl = (
                    exit_value
                    - position["Capital_Used"]
                    * (
                        1
                        + TRANSACTION_COST
                        + SLIPPAGE
                    )
                )

                net_return = (
                    exit_value
                    / (
                        position["Capital_Used"]
                        * (
                            1
                            + TRANSACTION_COST
                            + SLIPPAGE
                        )
                    )
                    - 1
                )

                capital += (
                    position["Capital_Used"]
                    + pnl
                )

                trades.append(
                    {
                        "Symbol": symbol,
                        "Entry_Date": position["Entry_Date"],
                        "Exit_Date": current_date,
                        "Entry_Price": entry_price,
                        "Exit_Price": current_price,
                        "Capital_Used": position["Capital_Used"],
                        "Net_PnL": pnl,
                        "Return": net_return,
                        "Days_Held": days_held,
                        "Exit_Reason": exit_reason,
                        "Buy_Probability": position[
                            "Buy_Probability"
                        ],
                        "Expected_Return_20D": position[
                            "Expected_Return_20D"
                        ],
                        "Technical_Score": position[
                            "Technical_Score"
                        ],
                        "Market_Regime": position[
                            "Market_Regime"
                        ],
                        "Rank_Score": position[
                            "Rank_Score"
                        ],
                    }
                )

            else:

                position["Current_Price"] = current_price

                remaining_positions.append(
                    position
                )

        open_positions = remaining_positions

        # ====================================================
        # CURRENT EQUITY
        # ====================================================

        open_value = 0

        for position in open_positions:

            current_price = position.get(
                "Current_Price",
                position["Entry_Price"]
            )

            return_now = (
                current_price
                / position["Entry_Price"]
                - 1
            )

            open_value += (
                position["Capital_Used"]
                * (1 + return_now)
            )

        equity = capital + open_value

        equity_curve.append(
            {
                "Date": current_date,
                "Equity": equity,
            }
        )

        # ====================================================
        # FIND TODAY'S CANDIDATES
        # ====================================================

        today = predictions[
            predictions["Date"] == current_date
        ].copy()

        today["Signal"] = today.apply(
            generate_signal,
            axis=1
        )

        today = today[
            today["Signal"]
        ].copy()

        if today.empty:
            continue

        today["Rank_Score"] = today.apply(
            calculate_rank,
            axis=1
        )

        today = today.sort_values(
            "Rank_Score",
            ascending=False
        )

        # ====================================================
        # EXISTING SYMBOLS
        # ====================================================

        existing_symbols = {
            p["Symbol"]
            for p in open_positions
        }

        today = today[
            ~today["Symbol"].isin(
                existing_symbols
            )
        ]

        # ====================================================
        # AVAILABLE POSITION SLOTS
        # ====================================================

        slots = (
            MAX_POSITIONS
            - len(open_positions)
        )

        if slots <= 0:
            continue

        today = today.head(slots)

        # ====================================================
        # CURRENT EXPOSURE
        # ====================================================

        current_exposure = sum(
            p["Capital_Used"]
            for p in open_positions
        )

        available_capital = max(
            0,
            capital
            * MAX_EXPOSURE
            - current_exposure
        )

        if available_capital <= 0:
            continue

        # ====================================================
        # ENTER POSITIONS
        # ====================================================

        for _, row in today.iterrows():

            regime = str(
                row["Market_Regime"]
            ).upper()

            rank = float(
                row["Rank_Score"]
            )

            # Same V4 sizing concept.
            if regime == "BEARISH":

                if rank >= 0.30:
                    allocation = 0.20
                else:
                    allocation = 0.15

            elif regime == "NEUTRAL":

                if rank >= 0.30:
                    allocation = 0.15
                else:
                    allocation = 0.10

            else:

                allocation = 0.10

            capital_used = min(
                capital * allocation,
                available_capital
            )

            if capital_used <= 0:
                continue

            symbol = row["Symbol"]

            if symbol not in market_data:
                continue

            market = market_data[symbol]

            available = market.loc[
                market.index >= current_date
            ]

            if available.empty:
                continue

            entry_price = float(
                available["Close_Price"].iloc[0]
            )

            capital -= capital_used

            position = {
                "Symbol": symbol,
                "Entry_Date": current_date,
                "Entry_Price": entry_price,
                "Capital_Used": capital_used,
                "Current_Price": entry_price,
                "Buy_Probability": row[
                    "Buy_Probability"
                ],
                "Expected_Return_20D": row[
                    "Expected_Return_20D"
                ],
                "Technical_Score": row[
                    "Technical_Score"
                ],
                "Market_Regime": row[
                    "Market_Regime"
                ],
                "Rank_Score": rank,
            }

            open_positions.append(position)

            available_capital -= capital_used

            if available_capital <= 0:
                break

    # ========================================================
    # FORCE CLOSE REMAINING POSITIONS
    # ========================================================

    if dates:

        final_date = dates[-1]

        for position in open_positions:

            symbol = position["Symbol"]

            if symbol not in market_data:
                continue

            market = market_data[symbol]

            available = market.loc[
                market.index <= final_date
            ]

            if available.empty:
                continue

            current_price = float(
                available["Close_Price"].iloc[-1]
            )

            entry_price = position["Entry_Price"]

            raw_return = (
                current_price
                / entry_price
                - 1
            )

            exit_value = (
                position["Capital_Used"]
                * (1 + raw_return)
                * (
                    1
                    - TRANSACTION_COST
                    - SLIPPAGE
                )
            )

            pnl = (
                exit_value
                - position["Capital_Used"]
                * (
                    1
                    + TRANSACTION_COST
                    + SLIPPAGE
                )
            )

            net_return = (
                exit_value
                / (
                    position["Capital_Used"]
                    * (
                        1
                        + TRANSACTION_COST
                        + SLIPPAGE
                    )
                )
                - 1
            )

            capital += (
                position["Capital_Used"]
                + pnl
            )

            trades.append(
                {
                    "Symbol": symbol,
                    "Entry_Date": position["Entry_Date"],
                    "Exit_Date": final_date,
                    "Entry_Price": entry_price,
                    "Exit_Price": current_price,
                    "Capital_Used": position["Capital_Used"],
                    "Net_PnL": pnl,
                    "Return": net_return,
                    "Days_Held": (
                        pd.Timestamp(final_date)
                        - pd.Timestamp(
                            position["Entry_Date"]
                        )
                    ).days,
                    "Exit_Reason": "END",
                    "Buy_Probability": position[
                        "Buy_Probability"
                    ],
                    "Expected_Return_20D": position[
                        "Expected_Return_20D"
                    ],
                    "Technical_Score": position[
                        "Technical_Score"
                    ],
                    "Market_Regime": position[
                        "Market_Regime"
                    ],
                    "Rank_Score": position[
                        "Rank_Score"
                    ],
                }
            )

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    return equity_df, trades_df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(equity_df, trades_df):

    if equity_df.empty:
        return {}

    equity = equity_df[
        "Equity"
    ].astype(float)

    start = equity.iloc[0]
    final = equity.iloc[-1]

    total_return = final / start - 1

    days = (
        pd.Timestamp(
            equity_df["Date"].iloc[-1]
        )
        - pd.Timestamp(
            equity_df["Date"].iloc[0]
        )
    ).days

    if days > 0:
        cagr = (
            final / start
        ) ** (365.25 / days) - 1
    else:
        cagr = np.nan

    daily_returns = equity.pct_change().dropna()

    if (
        len(daily_returns) > 1
        and daily_returns.std() > 0
    ):
        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    drawdown = (
        equity
        / equity.cummax()
        - 1
    )

    max_drawdown = drawdown.min()

    if trades_df.empty:
        win_rate = np.nan
        avg_trade = np.nan
        profit_factor = np.nan
    else:

        win_rate = (
            trades_df["Return"] > 0
        ).mean()

        avg_trade = (
            trades_df["Return"].mean()
        )

        gross_profit = trades_df.loc[
            trades_df["Net_PnL"] > 0,
            "Net_PnL"
        ].sum()

        gross_loss = abs(
            trades_df.loc[
                trades_df["Net_PnL"] < 0,
                "Net_PnL"
            ].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = np.inf

    return {
        "Final_Capital": final,
        "Return": total_return,
        "CAGR": cagr,
        "Max_Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Trades": len(trades_df),
        "Win_Rate": win_rate,
        "Average_Trade": avg_trade,
        "Profit_Factor": profit_factor,
    }


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V8 - MULTI-PERIOD WALK-FORWARD VALIDATION")
print("=" * 70)

print("\nLoading predictions...")

if not PREDICTION_FILE.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n"
        f"{PREDICTION_FILE}"
    )

predictions = pd.read_csv(
    PREDICTION_FILE
)

predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)

predictions = predictions.sort_values(
    ["Date", "Symbol"]
)

print(
    f"Predictions loaded: "
    f"{len(predictions):,}"
)


# ============================================================
# LOAD MARKET DATA
# ============================================================

symbols = sorted(
    predictions["Symbol"]
    .dropna()
    .unique()
)

print("\nLoading market data...")

market_data = load_market_data(
    symbols
)

print(
    f"Market datasets loaded: "
    f"{len(market_data)}/{len(symbols)}"
)


# ============================================================
# CREATE SEQUENTIAL PERIODS
# ============================================================

all_dates = sorted(
    predictions["Date"]
    .dropna()
    .unique()
)

date_count = len(all_dates)

period_size = date_count // N_PERIODS

periods = []

for i in range(N_PERIODS):

    start_idx = i * period_size

    if i == N_PERIODS - 1:
        end_idx = date_count
    else:
        end_idx = (
            (i + 1)
            * period_size
        )

    period_dates = all_dates[
        start_idx:end_idx
    ]

    if len(period_dates) == 0:
        continue

    periods.append(
        {
            "Period": i + 1,
            "Start": period_dates[0],
            "End": period_dates[-1],
        }
    )


# ============================================================
# RUN PERIOD TESTS
# ============================================================

all_period_results = []

all_trades = []

all_equity = []

for period in periods:

    period_number = period["Period"]

    start_date = period["Start"]
    end_date = period["End"]

    print("\n" + "=" * 70)

    print(
        f"PERIOD {period_number}"
    )

    print("=" * 70)

    print(
        f"\nDate range:"
        f"\n{pd.Timestamp(start_date).date()}"
        f" -> "
        f"{pd.Timestamp(end_date).date()}"
    )

    period_predictions = predictions[
        (
            predictions["Date"]
            >= start_date
        )
        &
        (
            predictions["Date"]
            <= end_date
        )
    ].copy()

    print(
        f"Prediction rows: "
        f"{len(period_predictions):,}"
    )

    equity_df, trades_df = run_backtest(
        period_predictions,
        market_data
    )

    metrics = calculate_metrics(
        equity_df,
        trades_df
    )

    metrics["Period"] = period_number
    metrics["Start_Date"] = start_date
    metrics["End_Date"] = end_date

    all_period_results.append(
        metrics
    )

    if not trades_df.empty:

        trades_df["Period"] = (
            period_number
        )

        all_trades.append(
            trades_df
        )

    if not equity_df.empty:

        equity_df["Period"] = (
            period_number
        )

        all_equity.append(
            equity_df
        )

    print(
        f"\nFinal Capital: "
        f"₹{metrics['Final_Capital']:,.2f}"
    )

    print(
        f"Return: "
        f"{metrics['Return']:.2%}"
    )

    print(
        f"CAGR: "
        f"{metrics['CAGR']:.2%}"
    )

    print(
        f"Max Drawdown: "
        f"{metrics['Max_Drawdown']:.2%}"
    )

    print(
        f"Sharpe: "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Trades: "
        f"{metrics['Trades']}"
    )

    print(
        f"Win Rate: "
        f"{metrics['Win_Rate']:.2%}"
    )

    print(
        f"Average Trade: "
        f"{metrics['Average_Trade']:.2%}"
    )

    print(
        f"Profit Factor: "
        f"{metrics['Profit_Factor']:.2f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

print_section = (
    lambda title:
    print(
        "\n"
        + "=" * 70
        + "\n"
        + title
        + "\n"
        + "=" * 70
    )
)

print_section(
    "MULTI-PERIOD RESULTS"
)

results_df = pd.DataFrame(
    all_period_results
)

results_df = results_df[
    [
        "Period",
        "Start_Date",
        "End_Date",
        "Final_Capital",
        "Return",
        "CAGR",
        "Max_Drawdown",
        "Sharpe",
        "Trades",
        "Win_Rate",
        "Average_Trade",
        "Profit_Factor",
    ]
]

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

print_section(
    "ROBUSTNESS SUMMARY"
)

positive_periods = (
    results_df["Return"] > 0
).sum()

profitable_period_pct = (
    positive_periods
    / len(results_df)
)

average_return = (
    results_df["Return"].mean()
)

median_return = (
    results_df["Return"].median()
)

average_sharpe = (
    results_df["Sharpe"].mean()
)

average_drawdown = (
    results_df["Max_Drawdown"].mean()
)

total_trades = (
    results_df["Trades"].sum()
)

print(
    f"\nProfitable periods: "
    f"{positive_periods}/"
    f"{len(results_df)}"
)

print(
    f"Profitable period rate: "
    f"{profitable_period_pct:.2%}"
)

print(
    f"Average period return: "
    f"{average_return:.2%}"
)

print(
    f"Median period return: "
    f"{median_return:.2%}"
)

print(
    f"Average Sharpe: "
    f"{average_sharpe:.2f}"
)

print(
    f"Average max drawdown: "
    f"{average_drawdown:.2%}"
)

print(
    f"Total trades: "
    f"{total_trades}"
)


# ============================================================
# COMBINED TRADE ANALYSIS
# ============================================================

if all_trades:

    combined_trades = pd.concat(
        all_trades,
        ignore_index=True
    )

else:

    combined_trades = pd.DataFrame()


if not combined_trades.empty:

    combined_win_rate = (
        combined_trades["Return"] > 0
    ).mean()

    combined_avg_trade = (
        combined_trades["Return"].mean()
    )

    gross_profit = combined_trades.loc[
        combined_trades["Net_PnL"] > 0,
        "Net_PnL"
    ].sum()

    gross_loss = abs(
        combined_trades.loc[
            combined_trades["Net_PnL"] < 0,
            "Net_PnL"
        ].sum()
    )

    combined_profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    print_section(
        "COMBINED TRADE RESULTS"
    )

    print(
        f"\nTotal trades: "
        f"{len(combined_trades)}"
    )

    print(
        f"Win rate: "
        f"{combined_win_rate:.2%}"
    )

    print(
        f"Average trade: "
        f"{combined_avg_trade:.2%}"
    )

    print(
        f"Profit factor: "
        f"{combined_profit_factor:.2f}"
    )

    combined_trades.to_csv(
        OUTPUT_DIR
        / "all_period_trades.csv",
        index=False
    )


# ============================================================
# SAVE EQUITY
# ============================================================

if all_equity:

    combined_equity = pd.concat(
        all_equity,
        ignore_index=True
    )

    combined_equity.to_csv(
        OUTPUT_DIR
        / "all_period_equity.csv",
        index=False
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_DIR
    / "period_results.csv",
    index=False
)

robustness_summary = pd.DataFrame(
    [
        {
            "Periods": len(results_df),
            "Profitable_Periods": positive_periods,
            "Profitable_Period_Rate":
                profitable_period_pct,
            "Average_Period_Return":
                average_return,
            "Median_Period_Return":
                median_return,
            "Average_Sharpe":
                average_sharpe,
            "Average_Max_Drawdown":
                average_drawdown,
            "Total_Trades":
                total_trades,
        }
    ]
)

robustness_summary.to_csv(
    OUTPUT_DIR
    / "robustness_summary.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print_section(
    "V8 MULTI-PERIOD VALIDATION COMPLETE"
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFiles created:")

print("  period_results.csv")
print("  robustness_summary.csv")
print("  all_period_trades.csv")
print("  all_period_equity.csv")

print("\n" + "=" * 70)