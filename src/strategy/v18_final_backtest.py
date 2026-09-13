from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SIGNAL_FILE = Path(
    "data/models/v17_signal_filter/v17_signals.csv"
)

FEATURE_DIR = Path(
    "data/processed"
)

OUTPUT_DIR = Path(
    "data/models/v18_final_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000.0

MAX_POSITIONS = 4

MAX_EXPOSURE = 0.80

POSITION_SIZE = 0.20

STOP_LOSS = 0.06

TARGET = 0.12

MAX_HOLD_DAYS = 20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(
    df,
    column="Date"
):

    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    if (
        hasattr(
            df[column].dt,
            "tz"
        )
        and df[column].dt.tz is not None
    ):

        df[column] = (
            df[column]
            .dt.tz_localize(None)
        )

    df[column] = (
        df[column]
        .dt.normalize()
    )

    return df


# ============================================================
# LOAD FEATURE FILES
# ============================================================

def load_market_data():

    market_data = {}

    symbols = sorted(
        pd.read_csv(
            SIGNAL_FILE,
            usecols=["Symbol"]
        )[
            "Symbol"
        ]
        .dropna()
        .unique()
    )

    for symbol in symbols:

        filename = (
            symbol
            .replace(
                ".",
                "_"
            )
            + "_features.csv"
        )

        path = (
            FEATURE_DIR
            / filename
        )

        if not path.exists():

            print(
                f"WARNING: missing {symbol}"
            )

            continue

        df = pd.read_csv(
            path
        )

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        market_data[
            symbol
        ] = df

    return market_data


# ============================================================
# EXECUTION PRICE
# ============================================================

def apply_buy_cost(
    price
):

    return price * (
        1
        + SLIPPAGE
    )


def apply_sell_cost(
    price
):

    return price * (
        1
        - SLIPPAGE
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    signals,
    market_data,
    start_date,
    end_date
):

    dates = sorted(
        signals[
            "Date"
        ]
        .unique()
    )

    dates = [
        d for d in dates
        if (
            d >= start_date
            and d <= end_date
        )
    ]

    cash = INITIAL_CAPITAL

    positions = []

    trades = []

    equity_rows = []

    # --------------------------------------------------------
    # Daily loop
    # --------------------------------------------------------

    for current_date in dates:

        # ====================================================
        # 1. PROCESS EXISTING POSITIONS
        # ====================================================

        remaining_positions = []

        for position in positions:

            symbol = position[
                "Symbol"
            ]

            entry_price = position[
                "Entry_Price"
            ]

            entry_date = position[
                "Entry_Date"
            ]

            shares = position[
                "Shares"
            ]

            capital_used = position[
                "Capital"
            ]

            df = market_data.get(
                symbol
            )

            if df is None:
                remaining_positions.append(
                    position
                )
                continue

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:

                remaining_positions.append(
                    position
                )
                continue

            row = rows.iloc[0]

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            close = float(
                row["Close"]
            )

            days_held = (
                current_date
                - entry_date
            ).days

            stop_price = (
                entry_price
                * (1 - STOP_LOSS)
            )

            target_price = (
                entry_price
                * (1 + TARGET)
            )

            exit_reason = None

            exit_price = None

            # =================================================
            # STOP / TARGET
            # =================================================
            #
            # If both happen on the same day we use STOP.
            # This is conservative because daily OHLC does
            # not tell us which happened first.
            # =================================================

            if low <= stop_price:

                exit_price = (
                    stop_price
                )

                exit_reason = "STOP"

            elif high >= target_price:

                exit_price = (
                    target_price
                )

                exit_reason = "TARGET"

            elif (
                days_held
                >= MAX_HOLD_DAYS
            ):

                exit_price = close

                exit_reason = "TIME"

            # -------------------------------------------------
            # End of available data
            # -------------------------------------------------

            elif current_date == end_date:

                exit_price = close

                exit_reason = "END"

            # -------------------------------------------------
            # Still holding
            # -------------------------------------------------

            if exit_reason is None:

                remaining_positions.append(
                    position
                )

                continue

            # =================================================
            # SELL
            # =================================================

            execution_price = (
                apply_sell_cost(
                    exit_price
                )
            )

            gross_value = (
                shares
                * execution_price
            )

            sell_cost = (
                gross_value
                * TRANSACTION_COST
            )

            net_value = (
                gross_value
                - sell_cost
            )

            pnl = (
                net_value
                - capital_used
            )

            return_pct = (
                pnl
                / capital_used
                * 100
            )

            cash += net_value

            trades.append({

                "Symbol":
                    symbol,

                "Entry_Date":
                    entry_date,

                "Exit_Date":
                    current_date,

                "Entry_Price":
                    entry_price,

                "Exit_Price":
                    exit_price,

                "Shares":
                    shares,

                "Capital":
                    capital_used,

                "PnL":
                    pnl,

                "Return_Pct":
                    return_pct,

                "Days_Held":
                    days_held,

                "Exit_Reason":
                    exit_reason,

                "Expected_Return_20D":
                    position[
                        "Expected_Return_20D"
                    ],

                "Quality_Score":
                    position[
                        "Quality_Score"
                    ],
            })

        positions = (
            remaining_positions
        )

        # ====================================================
        # 2. NEW BUY SIGNALS
        # ====================================================

        todays_signals = signals[
            signals["Date"]
            == current_date
        ]

        candidates = todays_signals[
            todays_signals[
                "Signal"
            ]
            == "BUY"
        ].copy()

        # Already-held symbols

        held_symbols = {
            p["Symbol"]
            for p in positions
        }

        candidates = candidates[
            ~candidates[
                "Symbol"
            ].isin(
                held_symbols
            )
        ]

        # ====================================================
        # RANK BUY SIGNALS
        # ====================================================

        if not candidates.empty:

            candidates[
                "Rank"
            ] = (

                candidates[
                    "Quality_Score"
                ]
                * 0.50

                + (
                    candidates[
                        "Expected_Return_20D"
                    ]
                    / 20
                )
                * 0.30

                + (
                    candidates[
                        "Technical_Score"
                    ]
                    / 4
                )
                * 0.20
            )

            candidates = (
                candidates
                .sort_values(
                    "Rank",
                    ascending=False
                )
            )

        # ====================================================
        # OPEN POSITIONS
        # ====================================================

        available_slots = (
            MAX_POSITIONS
            - len(positions)
        )

        current_exposure = sum(
            p["Capital"]
            for p in positions
        )

        maximum_new_exposure = (
            INITIAL_CAPITAL
            * MAX_EXPOSURE
            - current_exposure
        )

        if (
            available_slots > 0
            and maximum_new_exposure > 0
            and not candidates.empty
        ):

            for _, signal in (
                candidates.iterrows()
            ):

                if available_slots <= 0:
                    break

                if (
                    maximum_new_exposure
                    <= 0
                ):
                    break

                symbol = signal[
                    "Symbol"
                ]

                df = market_data.get(
                    symbol
                )

                if df is None:
                    continue

                rows = df[
                    df["Date"]
                    == current_date
                ]

                if rows.empty:
                    continue

                row = rows.iloc[0]

                close = float(
                    row["Close"]
                )

                entry_price = (
                    apply_buy_cost(
                        close
                    )
                )

                capital = min(

                    INITIAL_CAPITAL
                    * POSITION_SIZE,

                    maximum_new_exposure,

                    cash
                )

                if capital <= 0:
                    continue

                buy_cost = (
                    capital
                    * TRANSACTION_COST
                )

                total_required = (
                    capital
                    + buy_cost
                )

                if total_required > cash:
                    continue

                shares = (
                    capital
                    / entry_price
                )

                cash -= total_required

                positions.append({

                    "Symbol":
                        symbol,

                    "Entry_Date":
                        current_date,

                    "Entry_Price":
                        entry_price,

                    "Shares":
                        shares,

                    "Capital":
                        capital,

                    "Expected_Return_20D":
                        float(
                            signal[
                                "Expected_Return_20D"
                            ]
                        ),

                    "Quality_Score":
                        float(
                            signal[
                                "Quality_Score"
                            ]
                        ),
                })

                available_slots -= 1

                maximum_new_exposure -= (
                    capital
                )

        # ====================================================
        # 3. PORTFOLIO EQUITY
        # ====================================================

        position_value = 0

        for position in positions:

            symbol = position[
                "Symbol"
            ]

            df = market_data.get(
                symbol
            )

            if df is None:
                continue

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:
                continue

            close = float(
                rows.iloc[0]["Close"]
            )

            position_value += (
                position["Shares"]
                * close
            )

        equity = (
            cash
            + position_value
        )

        equity_rows.append({

            "Date":
                current_date,

            "Cash":
                cash,

            "Position_Value":
                position_value,

            "Equity":
                equity,

            "Open_Positions":
                len(positions),
        })

    # ========================================================
    # FORCE CLOSE REMAINING POSITIONS
    # ========================================================

    if positions:

        final_date = dates[-1]

        for position in positions:

            symbol = position[
                "Symbol"
            ]

            df = market_data.get(
                symbol
            )

            if df is None:
                continue

            rows = df[
                df["Date"]
                == final_date
            ]

            if rows.empty:
                continue

            close = float(
                rows.iloc[0]["Close"]
            )

            exit_price = close

            execution_price = (
                apply_sell_cost(
                    exit_price
                )
            )

            shares = position[
                "Shares"
            ]

            gross_value = (
                shares
                * execution_price
            )

            sell_cost = (
                gross_value
                * TRANSACTION_COST
            )

            net_value = (
                gross_value
                - sell_cost
            )

            capital_used = position[
                "Capital"
            ]

            pnl = (
                net_value
                - capital_used
            )

            return_pct = (
                pnl
                / capital_used
                * 100
            )

            cash += net_value

            trades.append({

                "Symbol":
                    symbol,

                "Entry_Date":
                    position[
                        "Entry_Date"
                    ],

                "Exit_Date":
                    final_date,

                "Entry_Price":
                    position[
                        "Entry_Price"
                    ],

                "Exit_Price":
                    exit_price,

                "Shares":
                    shares,

                "Capital":
                    capital_used,

                "PnL":
                    pnl,

                "Return_Pct":
                    return_pct,

                "Days_Held":
                    (
                        final_date
                        - position[
                            "Entry_Date"
                        ]
                    ).days,

                "Exit_Reason":
                    "END",

                "Expected_Return_20D":
                    position[
                        "Expected_Return_20D"
                    ],

                "Quality_Score":
                    position[
                        "Quality_Score"
                    ],
            })

    return (
        pd.DataFrame(equity_rows),
        pd.DataFrame(trades)
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    equity,
    trades
):

    if equity.empty:
        return {}

    equity = equity.copy()

    equity["Date"] = pd.to_datetime(
        equity["Date"]
    )

    equity = (
        equity
        .sort_values("Date")
        .reset_index(drop=True)
    )

    initial = float(
        equity.iloc[0]["Equity"]
    )

    final = float(
        equity.iloc[-1]["Equity"]
    )

    total_return = (
        final
        / initial
        - 1
    )

    days = max(
        (
            equity.iloc[-1]["Date"]
            - equity.iloc[0]["Date"]
        ).days,
        1
    )

    years = (
        days / 365.25
    )

    cagr = (
        (
            final / initial
        ) ** (
            1 / years
        )
        - 1
        if final > 0
        else -1
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    equity["Peak"] = (
        equity["Equity"]
        .cummax()
    )

    equity["Drawdown"] = (
        equity["Equity"]
        / equity["Peak"]
        - 1
    )

    max_drawdown = float(
        equity[
            "Drawdown"
        ].min()
    )

    # --------------------------------------------------------
    # Daily returns
    # --------------------------------------------------------

    daily_returns = (
        equity[
            "Equity"
        ]
        .pct_change()
        .dropna()
    )

    if (
        len(daily_returns) > 1
        and daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )

        volatility = (
            daily_returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

        volatility = 0

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    trade_count = len(
        trades
    )

    if trade_count > 0:

        wins = trades[
            trades["PnL"] > 0
        ]

        losses = trades[
            trades["PnL"] < 0
        ]

        win_rate = (
            len(wins)
            / trade_count
        )

        average_trade = (
            trades[
                "Return_Pct"
            ].mean()
        )

        gross_profit = (
            wins["PnL"].sum()
            if not wins.empty
            else 0
        )

        gross_loss = abs(
            losses["PnL"].sum()
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else np.inf
        )

    else:

        win_rate = 0

        average_trade = 0

        profit_factor = 0

    return {

        "Initial_Capital":
            initial,

        "Final_Capital":
            final,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Volatility":
            volatility,

        "Trades":
            trade_count,

        "Win_Rate":
            win_rate,

        "Average_Trade":
            average_trade,

        "Profit_Factor":
            profit_factor,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "V18 - FINAL RISK-MANAGED BACKTEST"
    )
    print("=" * 70)

    # ========================================================
    # LOAD SIGNALS
    # ========================================================

    signals = pd.read_csv(
        SIGNAL_FILE
    )

    signals = normalize_dates(
        signals
    )

    signals = (
        signals
        .dropna(
            subset=[
                "Date",
                "Symbol",
                "Signal"
            ]
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nSignals: "
        f"{len(signals):,}"
    )

    # ========================================================
    # MARKET DATA
    # ========================================================

    print(
        "\nLoading market data..."
    )

    market_data = (
        load_market_data()
    )

    print(
        f"Market datasets: "
        f"{len(market_data)}"
    )

    if not market_data:

        raise RuntimeError(
            "No market data found."
        )

    # ========================================================
    # VALID DATE RANGE
    # ========================================================

    start_date = (
        signals["Date"].min()
    )

    end_date = (
        signals["Date"].max()
    )

    print(
        f"\nBacktest period:"
    )

    print(
        f"{start_date.date()} "
        f"-> "
        f"{end_date.date()}"
    )

    # ========================================================
    # RUN
    # ========================================================

    equity, trades = run_backtest(

        signals,

        market_data,

        start_date,

        end_date
    )

    if equity.empty:

        raise RuntimeError(
            "Backtest produced no equity data."
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        equity,
        trades
    )

    # ========================================================
    # SAVE
    # ========================================================

    equity_file = (
        OUTPUT_DIR
        / "equity_curve.csv"
    )

    trades_file = (
        OUTPUT_DIR
        / "trades.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "summary.csv"
    )

    equity.to_csv(
        equity_file,
        index=False
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V18 RESULTS"
    )
    print("=" * 70)

    print(
        f"Initial Capital : "
        f"₹{metrics['Initial_Capital']:,.2f}"
    )

    print(
        f"Final Capital   : "
        f"₹{metrics['Final_Capital']:,.2f}"
    )

    print(
        f"Total Return    : "
        f"{metrics['Total_Return'] * 100:.2f}%"
    )

    print(
        f"CAGR            : "
        f"{metrics['CAGR'] * 100:.2f}%"
    )

    print(
        f"Max Drawdown    : "
        f"{metrics['Max_Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe          : "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Volatility      : "
        f"{metrics['Volatility'] * 100:.2f}%"
    )

    print(
        f"Trades          : "
        f"{metrics['Trades']}"
    )

    print(
        f"Win Rate        : "
        f"{metrics['Win_Rate'] * 100:.2f}%"
    )

    print(
        f"Average Trade   : "
        f"{metrics['Average_Trade']:.2f}%"
    )

    print(
        f"Profit Factor   : "
        f"{metrics['Profit_Factor']:.2f}"
    )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    if not trades.empty:

        print("\n")
        print("=" * 70)
        print(
            "EXIT REASONS"
        )
        print("=" * 70)

        exit_summary = (
            trades
            .groupby(
                "Exit_Reason"
            )
            .agg(
                Trades=(
                    "PnL",
                    "count"
                ),
                PnL=(
                    "PnL",
                    "sum"
                ),
                Avg_Return=(
                    "Return_Pct",
                    "mean"
                ),
                Win_Rate=(
                    "PnL",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                )
            )
            .sort_values(
                "PnL",
                ascending=False
            )
        )

        print(
            exit_summary.to_string()
        )

        # ----------------------------------------------------
        # Stock analysis
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print(
            "STOCK PERFORMANCE"
        )
        print("=" * 70)

        stock_summary = (
            trades
            .groupby(
                "Symbol"
            )
            .agg(
                Trades=(
                    "PnL",
                    "count"
                ),
                PnL=(
                    "PnL",
                    "sum"
                ),
                Avg_Return=(
                    "Return_Pct",
                    "mean"
                ),
                Win_Rate=(
                    "PnL",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                )
            )
            .sort_values(
                "PnL",
                ascending=False
            )
        )

        print(
            stock_summary.to_string()
        )

        stock_summary.to_csv(
            OUTPUT_DIR
            / "stock_performance.csv"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V18 COMPLETE"
    )
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        equity_file.resolve()
    )

    print(
        trades_file.resolve()
    )

    print(
        summary_file.resolve()
    )


if __name__ == "__main__":
    main()