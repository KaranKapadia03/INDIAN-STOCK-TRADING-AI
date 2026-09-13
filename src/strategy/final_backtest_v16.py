from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

PREDICTION_FILE = Path(
    "data/models/v14_regression/v14_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/models/v16_final_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_CAPITAL = 100_000

MAX_POSITIONS = 4
MAX_EXPOSURE = 0.80
POSITION_SIZE = 0.20

STOP_LOSS = 0.06
TAKE_PROFIT = 0.12
MAX_HOLD_DAYS = 20

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

REGRESSION_WEIGHT = 0.40
TECHNICAL_WEIGHT = 0.25
MARKET_WEIGHT = 0.20
NEWS_WEIGHT = 0.15


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(df, column="Date"):
    """
    Convert dates to timezone-naive pandas timestamps.

    This prevents merge errors between:
        datetime64[us, UTC+05:30]
    and
        datetime64[us]
    """

    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    # If timezone-aware, remove timezone
    if hasattr(
        df[column].dt,
        "tz"
    ) and df[column].dt.tz is not None:

        df[column] = (
            df[column]
            .dt.tz_localize(None)
        )

    # Normalize to trading-day date
    df[column] = (
        df[column]
        .dt.normalize()
    )

    return df


# ============================================================
# FEATURE FILE
# ============================================================

def find_feature_file(symbol):

    filename = (
        symbol.replace(
            ".",
            "_"
        )
        + "_features.csv"
    )

    path = FEATURE_DIR / filename

    if path.exists():
        return path

    for file_path in FEATURE_DIR.glob(
        "*_features.csv"
    ):

        name = file_path.stem

        name = name.replace(
            "_features",
            ""
        )

        if name.endswith("_NS"):

            name = (
                name[:-3]
                + ".NS"
            )

        if name.upper() == symbol.upper():

            return file_path

    return None


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(row):

    score = 0

    close = row.get(
        "Close",
        np.nan
    )

    sma20 = row.get(
        "SMA_20",
        np.nan
    )

    sma50 = row.get(
        "SMA_50",
        np.nan
    )

    # Price vs SMA20

    if (
        pd.notna(close)
        and pd.notna(sma20)
    ):

        if close > sma20:
            score += 1
        else:
            score -= 1

    # SMA20 vs SMA50

    if (
        pd.notna(sma20)
        and pd.notna(sma50)
    ):

        if sma20 > sma50:
            score += 1
        else:
            score -= 1

    # MACD

    macd = row.get(
        "MACD",
        np.nan
    )

    macd_signal = row.get(
        "MACD_Signal",
        np.nan
    )

    if (
        pd.notna(macd)
        and pd.notna(macd_signal)
    ):

        if macd > macd_signal:
            score += 1
        else:
            score -= 1

    # RSI

    rsi = row.get(
        "RSI_14",
        np.nan
    )

    if pd.notna(rsi):

        if 50 <= rsi <= 70:

            score += 1

        elif rsi < 30:

            score += 1

        elif rsi > 75:

            score -= 1

    return score


# ============================================================
# MARKET SCORE
# ============================================================

def market_score(row):

    score = 0
    signals = 0

    # NIFTY 20D return

    nifty_return = row.get(
        "NIFTY_Return_20D",
        np.nan
    )

    if pd.notna(nifty_return):

        signals += 1

        if nifty_return > 2:

            score += 1

        elif nifty_return < -2:

            score -= 1

    # NIFTY SMA50 vs SMA200

    nifty_trend = row.get(
        "NIFTY_SMA50_vs_SMA200",
        np.nan
    )

    if pd.notna(nifty_trend):

        signals += 1

        if nifty_trend > 0:

            score += 1

        elif nifty_trend < 0:

            score -= 1

    # BANKNIFTY 20D return

    bank_return = row.get(
        "BANKNIFTY_Return_20D",
        np.nan
    )

    if pd.notna(bank_return):

        signals += 1

        if bank_return > 2:

            score += 1

        elif bank_return < -2:

            score -= 1

    if signals == 0:

        return 0.0

    return float(
        np.clip(
            score / signals,
            -1,
            1
        )
    )


# ============================================================
# HISTORICAL SIGNAL
# ============================================================

def calculate_signal(
    row,
    expected_return
):

    technical_raw = technical_score(
        row
    )

    technical = np.clip(
        technical_raw / 4.0,
        -1,
        1
    )

    market = market_score(
        row
    )

    # IMPORTANT:
    # Historical news is deliberately excluded.
    news = 0.0

    regression_component = np.tanh(
        expected_return / 5.0
    )

    final_score = (
        regression_component
        * REGRESSION_WEIGHT

        + technical
        * TECHNICAL_WEIGHT

        + market
        * MARKET_WEIGHT

        + news
        * NEWS_WEIGHT
    )

    final_score = float(
        np.clip(
            final_score,
            -1,
            1
        )
    )

    # BUY

    if (
        final_score >= 0.30
        and expected_return >= 2
    ):

        signal = "BUY"

    # SELL

    elif (
        final_score <= -0.30
        and expected_return <= -2
    ):

        signal = "SELL"

    # HOLD

    else:

        signal = "HOLD"

    return (
        signal,
        final_score,
        technical_raw,
        market
    )


# ============================================================
# LOAD DATA
# ============================================================

def prepare_data():

    # --------------------------------------------------------
    # V14 PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions = normalize_dates(
        predictions
    )

    # --------------------------------------------------------
    # V14 RESULTS
    # --------------------------------------------------------

    results_file = (
        PREDICTION_FILE.parent
        / "v14_model_results.csv"
    )

    results = pd.read_csv(
        results_file
    )

    # --------------------------------------------------------
    # SELECT BEST MODEL BY RMSE
    # --------------------------------------------------------

    best_models = (
        results
        .sort_values(
            [
                "Symbol",
                "RMSE"
            ]
        )
        .groupby(
            "Symbol"
        )
        .first()
        .reset_index()
    )

    best_model_map = dict(
        zip(
            best_models["Symbol"],
            best_models["Model"]
        )
    )

    predictions["Best_Model"] = (
        predictions["Symbol"]
        .map(best_model_map)
    )

    predictions = predictions[
        predictions["Model"]
        == predictions["Best_Model"]
    ].copy()

    # --------------------------------------------------------
    # STOCK DATA
    # --------------------------------------------------------

    all_data = {}

    symbols = sorted(
        predictions[
            "Symbol"
        ]
        .dropna()
        .unique()
    )

    for symbol in symbols:

        feature_file = (
            find_feature_file(
                symbol
            )
        )

        if feature_file is None:

            print(
                f"Missing feature file: "
                f"{symbol}"
            )

            continue

        df = pd.read_csv(
            feature_file
        )

        # FIX TIMEZONE HERE

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates(
                subset=["Date"]
            )
            .reset_index(drop=True)
        )

        stock_predictions = (
            predictions[
                predictions["Symbol"]
                == symbol
            ][
                [
                    "Date",
                    "Predicted_Return_20D"
                ]
            ]
            .sort_values("Date")
            .drop_duplicates(
                subset=["Date"]
            )
        )

        # FIX TIMEZONE ON PREDICTIONS TOO

        stock_predictions = normalize_dates(
            stock_predictions
        )

        # ----------------------------------------------------
        # SAFE MERGE
        # ----------------------------------------------------

        df = pd.merge(
            df,
            stock_predictions,
            on="Date",
            how="left"
        )

        df["Symbol"] = symbol

        all_data[
            symbol
        ] = df

    return all_data


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(data):

    cash = INITIAL_CAPITAL

    positions = {}

    trades = []

    equity_curve = []

    # --------------------------------------------------------
    # ALL TRADING DATES
    # --------------------------------------------------------

    dates = sorted(
        set(
            date
            for df in data.values()
            for date in df["Date"]
            if pd.notna(date)
        )
    )

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for current_date in dates:

        # ====================================================
        # EXIT POSITIONS
        # ====================================================

        for symbol in list(
            positions.keys()
        ):

            position = positions[
                symbol
            ]

            df = data[
                symbol
            ]

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            current_price = float(
                row["Close"]
            )

            entry_price = (
                position[
                    "entry_price"
                ]
            )

            entry_date = (
                position[
                    "entry_date"
                ]
            )

            days_held = (
                current_date
                - entry_date
            ).days

            return_pct = (
                current_price
                / entry_price
                - 1
            )

            exit_reason = None

            # Stop loss

            if return_pct <= -STOP_LOSS:

                exit_reason = "STOP"

            # Take profit

            elif return_pct >= TAKE_PROFIT:

                exit_reason = "TARGET"

            # Time exit

            elif days_held >= MAX_HOLD_DAYS:

                exit_reason = "TIME"

            if exit_reason is None:
                continue

            # ------------------------------------------------
            # SELL EXECUTION
            # ------------------------------------------------

            exit_price = (
                current_price
                * (
                    1
                    - SLIPPAGE
                )
            )

            gross_value = (
                position[
                    "shares"
                ]
                * exit_price
            )

            exit_cost = (
                gross_value
                * TRANSACTION_COST
            )

            net_value = (
                gross_value
                - exit_cost
            )

            cash += net_value

            pnl = (
                net_value
                - position[
                    "cost_basis"
                ]
            )

            trade_return = (
                net_value
                / position[
                    "cost_basis"
                ]
                - 1
            ) * 100

            trades.append({

                "Symbol":
                    symbol,

                "Entry_Date":
                    entry_date,

                "Exit_Date":
                    current_date,

                "Entry_Price":
                    position[
                        "entry_price"
                    ],

                "Exit_Price":
                    exit_price,

                "Shares":
                    position[
                        "shares"
                    ],

                "Days_Held":
                    days_held,

                "Return_Pct":
                    trade_return,

                "PnL":
                    pnl,

                "Exit_Reason":
                    exit_reason,

                "Final_Score":
                    position[
                        "final_score"
                    ],

                "Expected_Return":
                    position[
                        "expected_return"
                    ],

                "Technical_Score":
                    position[
                        "technical_score"
                    ],

                "Market_Score":
                    position[
                        "market_score"
                    ],
            })

            del positions[
                symbol
            ]

        # ====================================================
        # FIND BUY CANDIDATES
        # ====================================================

        candidates = []

        for symbol, df in data.items():

            if symbol in positions:
                continue

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            expected_return = row.get(
                "Predicted_Return_20D",
                np.nan
            )

            if pd.isna(
                expected_return
            ):
                continue

            (
                signal,
                final_score,
                tech_raw,
                market
            ) = calculate_signal(
                row,
                float(
                    expected_return
                )
            )

            if signal != "BUY":
                continue

            candidates.append({

                "Symbol":
                    symbol,

                "Date":
                    current_date,

                "Final_Score":
                    final_score,

                "Expected_Return":
                    float(
                        expected_return
                    ),

                "Technical_Score":
                    tech_raw,

                "Market_Score":
                    market,

                "Close":
                    float(
                        row["Close"]
                    ),
            })

        # ----------------------------------------------------
        # RANK CANDIDATES
        # ----------------------------------------------------

        candidates = sorted(
            candidates,
            key=lambda x: (
                x["Final_Score"],
                x["Expected_Return"]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # EXISTING EXPOSURE
        # ----------------------------------------------------

        current_equity = cash

        for symbol, position in positions.items():

            df = data[
                symbol
            ]

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:
                continue

            price = float(
                rows.iloc[0]["Close"]
            )

            current_equity += (
                position[
                    "shares"
                ]
                * price
            )

        current_exposure = 0

        for position in positions.values():

            current_exposure += (
                position[
                    "cost_basis"
                ]
                / max(
                    current_equity,
                    1
                )
            )

        available_slots = (
            MAX_POSITIONS
            - len(positions)
        )

        available_exposure = max(
            0,
            MAX_EXPOSURE
            - current_exposure
        )

        # ----------------------------------------------------
        # ENTER
        # ----------------------------------------------------

        for candidate in candidates:

            if available_slots <= 0:
                break

            if available_exposure <= 0:
                break

            position_fraction = min(
                POSITION_SIZE,
                available_exposure
            )

            allocation = (
                current_equity
                * position_fraction
            )

            if allocation <= 0:
                continue

            entry_price = (
                candidate["Close"]
                * (
                    1
                    + SLIPPAGE
                )
            )

            entry_cost = (
                allocation
                * TRANSACTION_COST
            )

            total_cost = (
                allocation
                + entry_cost
            )

            if total_cost > cash:
                continue

            shares = (
                allocation
                / entry_price
            )

            cash -= total_cost

            positions[
                candidate["Symbol"]
            ] = {

                "entry_date":
                    current_date,

                "entry_price":
                    entry_price,

                "shares":
                    shares,

                "cost_basis":
                    total_cost,

                "final_score":
                    candidate[
                        "Final_Score"
                    ],

                "expected_return":
                    candidate[
                        "Expected_Return"
                    ],

                "technical_score":
                    candidate[
                        "Technical_Score"
                    ],

                "market_score":
                    candidate[
                        "Market_Score"
                    ],
            }

            available_slots -= 1

            available_exposure -= (
                position_fraction
            )

        # ====================================================
        # END OF DAY EQUITY
        # ====================================================

        portfolio_value = cash

        for symbol, position in positions.items():

            df = data[
                symbol
            ]

            rows = df[
                df["Date"]
                == current_date
            ]

            if rows.empty:
                continue

            price = float(
                rows.iloc[0]["Close"]
            )

            portfolio_value += (
                position[
                    "shares"
                ]
                * price
            )

        equity_curve.append({

            "Date":
                current_date,

            "Equity":
                portfolio_value,

            "Cash":
                cash,

            "Positions":
                len(positions),
        })

    # ========================================================
    # CLOSE OPEN POSITIONS
    # ========================================================

    if positions and dates:

        final_date = dates[-1]

        for symbol in list(
            positions.keys()
        ):

            position = positions[
                symbol
            ]

            df = data[
                symbol
            ]

            rows = df[
                df["Date"]
                == final_date
            ]

            if rows.empty:
                continue

            current_price = float(
                rows.iloc[0]["Close"]
            )

            exit_price = (
                current_price
                * (
                    1
                    - SLIPPAGE
                )
            )

            gross_value = (
                position[
                    "shares"
                ]
                * exit_price
            )

            exit_cost = (
                gross_value
                * TRANSACTION_COST
            )

            net_value = (
                gross_value
                - exit_cost
            )

            cash += net_value

            pnl = (
                net_value
                - position[
                    "cost_basis"
                ]
            )

            trade_return = (
                net_value
                / position[
                    "cost_basis"
                ]
                - 1
            ) * 100

            days_held = (
                final_date
                - position[
                    "entry_date"
                ]
            ).days

            trades.append({

                "Symbol":
                    symbol,

                "Entry_Date":
                    position[
                        "entry_date"
                    ],

                "Exit_Date":
                    final_date,

                "Entry_Price":
                    position[
                        "entry_price"
                    ],

                "Exit_Price":
                    exit_price,

                "Shares":
                    position[
                        "shares"
                    ],

                "Days_Held":
                    days_held,

                "Return_Pct":
                    trade_return,

                "PnL":
                    pnl,

                "Exit_Reason":
                    "END",

                "Final_Score":
                    position[
                        "final_score"
                    ],

                "Expected_Return":
                    position[
                        "expected_return"
                    ],

                "Technical_Score":
                    position[
                        "technical_score"
                    ],

                "Market_Score":
                    position[
                        "market_score"
                    ],
            })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    if equity_df.empty:

        return (
            equity_df,
            trades_df,
            {}
        )

    # ========================================================
    # EQUITY METRICS
    # ========================================================

    equity_df["Peak"] = (
        equity_df["Equity"]
        .cummax()
    )

    equity_df["Drawdown"] = (
        equity_df["Equity"]
        / equity_df["Peak"]
        - 1
    ) * 100

    final_capital = float(
        equity_df[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    ) * 100

    max_drawdown = float(
        equity_df[
            "Drawdown"
        ].min()
    )

    days = (
        equity_df["Date"].iloc[-1]
        - equity_df["Date"].iloc[0]
    ).days

    years = max(
        days / 365.25,
        0.01
    )

    cagr = (
        (
            final_capital
            / INITIAL_CAPITAL
        )
        ** (
            1 / years
        )
        - 1
    ) * 100

    # ========================================================
    # TRADE METRICS
    # ========================================================

    if not trades_df.empty:

        win_rate = (
            trades_df[
                "Return_Pct"
            ] > 0
        ).mean() * 100

        avg_trade = (
            trades_df[
                "Return_Pct"
            ].mean()
        )

        gross_profit = trades_df.loc[
            trades_df["PnL"] > 0,
            "PnL"
        ].sum()

        gross_loss = abs(
            trades_df.loc[
                trades_df["PnL"] < 0,
                "PnL"
            ].sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

    else:

        win_rate = 0
        avg_trade = 0
        profit_factor = 0

    # ========================================================
    # SHARPE
    # ========================================================

    daily_returns = (
        equity_df[
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

    else:

        sharpe = 0

    metrics = {

        "Initial_Capital":
            INITIAL_CAPITAL,

        "Final_Capital":
            final_capital,

        "Total_Return_Pct":
            total_return,

        "CAGR_Pct":
            cagr,

        "Max_Drawdown_Pct":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Trades":
            len(trades_df),

        "Win_Rate_Pct":
            win_rate,

        "Average_Trade_Pct":
            avg_trade,

        "Profit_Factor":
            profit_factor,
    }

    return (
        equity_df,
        trades_df,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "V16 - FINAL SIGNAL BACKTEST"
    )
    print("=" * 70)

    print(
        "\nLoading data..."
    )

    data = prepare_data()

    print(
        f"Stocks loaded: "
        f"{len(data)}"
    )

    if not data:

        raise RuntimeError(
            "No stock data available."
        )

    # ========================================================
    # PERIOD
    # ========================================================

    all_dates = sorted(
        set(
            date
            for df in data.values()
            for date in df["Date"]
        )
    )

    start_date = all_dates[0]
    end_date = all_dates[-1]

    print(
        "\nBacktest period:"
    )

    print(
        f"{start_date.date()} "
        f"-> "
        f"{end_date.date()}"
    )

    # ========================================================
    # RUN
    # ========================================================

    (
        equity_df,
        trades_df,
        metrics
    ) = run_backtest(
        data
    )

    # ========================================================
    # SAVE
    # ========================================================

    equity_df.to_csv(
        OUTPUT_DIR
        / "equity_curve.csv",
        index=False
    )

    trades_df.to_csv(
        OUTPUT_DIR
        / "trades.csv",
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V16 BACKTEST RESULTS"
    )
    print("=" * 70)

    print(
        f"Initial Capital       "
        f"₹{metrics['Initial_Capital']:,.2f}"
    )

    print(
        f"Final Capital         "
        f"₹{metrics['Final_Capital']:,.2f}"
    )

    print(
        f"Total Return          "
        f"{metrics['Total_Return_Pct']:.2f}%"
    )

    print(
        f"CAGR                  "
        f"{metrics['CAGR_Pct']:.2f}%"
    )

    print(
        f"Max Drawdown          "
        f"{metrics['Max_Drawdown_Pct']:.2f}%"
    )

    print(
        f"Sharpe                "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Trades                "
        f"{metrics['Trades']}"
    )

    print(
        f"Win Rate              "
        f"{metrics['Win_Rate_Pct']:.2f}%"
    )

    print(
        f"Average Trade         "
        f"{metrics['Average_Trade_Pct']:.2f}%"
    )

    print(
        f"Profit Factor         "
        f"{metrics['Profit_Factor']:.2f}"
    )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    if not trades_df.empty:

        print("\n")
        print("=" * 70)
        print(
            "EXIT REASONS"
        )
        print("=" * 70)

        exit_stats = (
            trades_df
            .groupby(
                "Exit_Reason"
            )
            .agg(
                Trades=(
                    "Return_Pct",
                    "count"
                ),

                Avg_Return=(
                    "Return_Pct",
                    "mean"
                ),

                Win_Rate=(
                    "Return_Pct",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                    * 100
                ),

                PnL=(
                    "PnL",
                    "sum"
                )
            )
            .round(2)
        )

        print(
            exit_stats
        )

        # ----------------------------------------------------
        # STOCK PERFORMANCE
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print(
            "STOCK PERFORMANCE"
        )
        print("=" * 70)

        stock_stats = (
            trades_df
            .groupby(
                "Symbol"
            )
            .agg(
                Trades=(
                    "Return_Pct",
                    "count"
                ),

                Avg_Return=(
                    "Return_Pct",
                    "mean"
                ),

                Win_Rate=(
                    "Return_Pct",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                    * 100
                ),

                PnL=(
                    "PnL",
                    "sum"
                )
            )
            .sort_values(
                "PnL",
                ascending=False
            )
            .round(2)
        )

        print(
            stock_stats
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V16 COMPLETE"
    )
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()