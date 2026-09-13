from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
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
    / "strategy_v5_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100000.0

MAX_POSITIONS = 4

MAX_EXPOSURE = 0.80

STOP_LOSS = 0.06

TAKE_PROFIT = 0.12

MAX_HOLD_DAYS = 20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005


# ============================================================
# V4 RULES — KEPT UNCHANGED
# ============================================================

BEARISH_MIN_BUY_PROB = 0.55
BEARISH_MIN_TECHNICAL = 0
BEARISH_MIN_EXPECTED_RETURN = 0.0

NEUTRAL_MIN_BUY_PROB = 0.65
NEUTRAL_MIN_TECHNICAL = 1
NEUTRAL_MIN_EXPECTED_RETURN = 2.0

BULLISH_MIN_BUY_PROB = 0.75
BULLISH_MIN_TECHNICAL = 2
BULLISH_MIN_EXPECTED_RETURN = 4.0


# ============================================================
# VALIDATION SPLIT
# ============================================================

# First 60% = development
# Last 40% = completely unseen validation period

DEVELOPMENT_RATIO = 0.60


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V5 - OUT-OF-SAMPLE STRATEGY VALIDATION")
print("=" * 70)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading predictions...")

if not PREDICTION_FILE.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )


predictions = pd.read_csv(
    PREDICTION_FILE
)

predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)

predictions = predictions.sort_values(
    ["Date", "Symbol"]
).reset_index(
    drop=True
)


print(
    f"Predictions loaded: "
    f"{len(predictions):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Symbol",
    "Buy_Probability",
    "Sell_Probability",
    "Expected_Return_20D",
    "Technical_Score",
    "Market_Regime",
    "Final_Score",
]

missing = [
    col
    for col in required_columns
    if col not in predictions.columns
]

if missing:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# LOAD MARKET DATA
# ============================================================

print("\nLoading market data...")


def load_market_data(symbol):

    filename = (
        symbol.replace(".", "_")
        + ".csv"
    )

    path = (
        MARKET_DIR
        / filename
    )

    if not path.exists():

        return None

    df = pd.read_csv(
        path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    return df


market_data = {}

for symbol in sorted(
    predictions["Symbol"].unique()
):

    data = load_market_data(
        symbol
    )

    if data is not None:

        market_data[symbol] = data


print(
    f"Market datasets loaded: "
    f"{len(market_data)}/"
    f"{predictions['Symbol'].nunique()}"
)


# ============================================================
# PRICE HELPERS
# ============================================================

def get_price_row(symbol, date):

    df = market_data.get(symbol)

    if df is None:

        return None

    rows = df[
        df["Date"] == date
    ]

    if rows.empty:

        return None

    return rows.iloc[0]


def get_next_trading_day(symbol, date):

    df = market_data.get(symbol)

    if df is None:

        return None

    rows = df[
        df["Date"] > date
    ]

    if rows.empty:

        return None

    return rows.iloc[0]


# ============================================================
# V4 SIGNAL RULE
# ============================================================

def qualifies_for_v4(row):

    regime = str(
        row["Market_Regime"]
    ).upper()

    buy_probability = float(
        row["Buy_Probability"]
    )

    technical_score = float(
        row["Technical_Score"]
    )

    expected_return = float(
        row["Expected_Return_20D"]
    )


    # --------------------------------------------------------
    # BEARISH
    # --------------------------------------------------------

    if regime == "BEARISH":

        return (
            buy_probability
            >= BEARISH_MIN_BUY_PROB
            and
            technical_score
            >= BEARISH_MIN_TECHNICAL
            and
            expected_return
            >= BEARISH_MIN_EXPECTED_RETURN
        )


    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------

    if regime == "NEUTRAL":

        return (
            buy_probability
            >= NEUTRAL_MIN_BUY_PROB
            and
            technical_score
            >= NEUTRAL_MIN_TECHNICAL
            and
            expected_return
            >= NEUTRAL_MIN_EXPECTED_RETURN
        )


    # --------------------------------------------------------
    # BULLISH
    # --------------------------------------------------------

    if regime == "BULLISH":

        return (
            buy_probability
            >= BULLISH_MIN_BUY_PROB
            and
            technical_score
            >= BULLISH_MIN_TECHNICAL
            and
            expected_return
            >= BULLISH_MIN_EXPECTED_RETURN
        )


    return False


# ============================================================
# CREATE V4 SIGNAL
# ============================================================

predictions["V4_Candidate"] = predictions.apply(
    qualifies_for_v4,
    axis=1
)


predictions["V4_Rank_Score"] = (

    0.45
    * (
        predictions["Buy_Probability"]
        - predictions["Sell_Probability"]
    )

    +

    0.30
    * np.clip(
        predictions["Expected_Return_20D"]
        / 10.0,
        -1,
        1
    )

    +

    0.20
    * np.clip(
        predictions["Technical_Score"]
        / 4.0,
        -1,
        1
    )

    +

    0.05
    * np.where(
        predictions["Market_Regime"]
        == "BEARISH",
        0.15,
        np.where(
            predictions["Market_Regime"]
            == "BULLISH",
            0.05,
            0
        )
    )
)


# ============================================================
# DEVELOPMENT / VALIDATION SPLIT
# ============================================================

unique_dates = sorted(
    predictions["Date"].unique()
)

split_index = int(
    len(unique_dates)
    * DEVELOPMENT_RATIO
)

split_date = pd.Timestamp(
    unique_dates[split_index]
)


development = predictions[
    predictions["Date"]
    < split_date
].copy()


validation = predictions[
    predictions["Date"]
    >= split_date
].copy()


print(
    "\n"
    + "=" * 70
)

print(
    "DATA SPLIT"
)

print("=" * 70)


print(
    f"\nDevelopment period:"
)

print(
    f"{development['Date'].min().date()} "
    f"→ "
    f"{development['Date'].max().date()}"
)


print(
    f"Rows: "
    f"{len(development):,}"
)


print(
    f"\nValidation period:"
)

print(
    f"{validation['Date'].min().date()} "
    f"→ "
    f"{validation['Date'].max().date()}"
)


print(
    f"Rows: "
    f"{len(validation):,}"
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    period_name
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{period_name.upper()} BACKTEST"
    )

    print(
        "=" * 70
    )


    if data.empty:

        return {}


    cash = INITIAL_CAPITAL

    positions = {}

    equity_history = []

    trades = []


    dates = sorted(
        data["Date"].unique()
    )


    # ========================================================
    # DAILY LOOP
    # ========================================================

    for current_date in dates:

        current_date = pd.Timestamp(
            current_date
        )


        # ====================================================
        # MANAGE POSITIONS
        # ====================================================

        for symbol, position in list(
            positions.items()
        ):

            price_row = get_price_row(
                symbol,
                current_date
            )

            if price_row is None:

                continue


            high = float(
                price_row["High"]
            )

            low = float(
                price_row["Low"]
            )

            close = float(
                price_row["Close"]
            )


            position["last_price"] = close

            position["days_held"] += 1


            entry_price = (
                position["entry_price"]
            )


            stop_price = (
                entry_price
                * (1 - STOP_LOSS)
            )


            target_price = (
                entry_price
                * (1 + TAKE_PROFIT)
            )


            exit_reason = None

            exit_price = None


            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            if low <= stop_price:

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE)
                )

                exit_reason = "STOP"


            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            if (
                exit_reason is None
                and high >= target_price
            ):

                exit_price = (
                    target_price
                    * (1 - SLIPPAGE)
                )

                exit_reason = "TARGET"


            # ------------------------------------------------
            # TIME
            # ------------------------------------------------

            if (
                exit_reason is None
                and position["days_held"]
                >= MAX_HOLD_DAYS
            ):

                exit_price = (
                    close
                    * (1 - SLIPPAGE)
                )

                exit_reason = "TIME"


            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if exit_reason is not None:

                shares = position[
                    "shares"
                ]


                gross_value = (
                    shares
                    * exit_price
                )


                exit_cost = (
                    gross_value
                    * TRANSACTION_COST
                )


                cash += (
                    gross_value
                    - exit_cost
                )


                gross_pnl = (
                    shares
                    * (
                        exit_price
                        - entry_price
                    )
                )


                net_pnl = (
                    gross_pnl
                    - position[
                        "entry_cost"
                    ]
                    - exit_cost
                )


                trade_return = (
                    net_pnl
                    / position[
                        "capital_used"
                    ]
                )


                trades.append({

                    "Symbol":
                        symbol,

                    "Entry_Date":
                        position[
                            "entry_date"
                        ],

                    "Exit_Date":
                        current_date,

                    "Entry_Price":
                        entry_price,

                    "Exit_Price":
                        exit_price,

                    "Capital_Used":
                        position[
                            "capital_used"
                        ],

                    "Net_PnL":
                        net_pnl,

                    "Return":
                        trade_return,

                    "Days_Held":
                        position[
                            "days_held"
                        ],

                    "Exit_Reason":
                        exit_reason,

                    "Buy_Probability":
                        position[
                            "buy_probability"
                        ],

                    "Expected_Return_20D":
                        position[
                            "expected_return"
                        ],

                    "Technical_Score":
                        position[
                            "technical_score"
                        ],

                    "Market_Regime":
                        position[
                            "market_regime"
                        ],

                    "Rank_Score":
                        position[
                            "rank_score"
                        ]
                })


                del positions[symbol]


        # ====================================================
        # PORTFOLIO VALUE
        # ====================================================

        portfolio_value = cash


        for symbol, position in positions.items():

            price_row = get_price_row(
                symbol,
                current_date
            )


            if price_row is not None:

                price = float(
                    price_row["Close"]
                )

                position["last_price"] = (
                    price
                )

            else:

                price = position[
                    "last_price"
                ]


            portfolio_value += (
                position["shares"]
                * price
            )


        # ====================================================
        # EXPOSURE
        # ====================================================

        current_exposure = sum(

            position["shares"]
            * position["last_price"]

            for position
            in positions.values()
        )


        max_exposure = (
            portfolio_value
            * MAX_EXPOSURE
        )


        available_exposure = max(
            0,
            max_exposure
            - current_exposure
        )


        # ====================================================
        # CANDIDATES
        # ====================================================

        daily = data[
            data["Date"]
            == current_date
        ].copy()


        candidates = daily[
            daily["V4_Candidate"]
        ].copy()


        candidates = candidates.sort_values(
            "V4_Rank_Score",
            ascending=False
        )


        # ====================================================
        # ENTER
        # ====================================================

        selected = 0


        for _, candidate in candidates.iterrows():

            if selected >= MAX_POSITIONS:

                break


            symbol = candidate[
                "Symbol"
            ]


            if symbol in positions:

                continue


            if available_exposure <= 0:

                break


            next_row = get_next_trading_day(
                symbol,
                current_date
            )


            if next_row is None:

                continue


            entry_date = pd.Timestamp(
                next_row["Date"]
            )


            entry_price = float(
                next_row["Open"]
            )


            entry_price *= (
                1 + SLIPPAGE
            )


            # ------------------------------------------------
            # Regime-dependent sizing
            # ------------------------------------------------

            regime = str(
                candidate[
                    "Market_Regime"
                ]
            ).upper()


            rank_score = float(
                candidate[
                    "V4_Rank_Score"
                ]
            )


            if regime == "BEARISH":

                if rank_score >= 0.30:

                    position_fraction = 0.20

                else:

                    position_fraction = 0.15


            elif regime == "BULLISH":

                position_fraction = 0.10


            else:

                position_fraction = 0.10


            capital_to_use = min(

                portfolio_value
                * position_fraction,

                available_exposure,

                cash
            )


            if capital_to_use <= 0:

                continue


            entry_cost = (
                capital_to_use
                * TRANSACTION_COST
            )


            total_required = (
                capital_to_use
                + entry_cost
            )


            if total_required > cash:

                continue


            shares = (
                capital_to_use
                / entry_price
            )


            cash -= total_required


            positions[symbol] = {

                "entry_date":
                    entry_date,

                "entry_price":
                    entry_price,

                "shares":
                    shares,

                "capital_used":
                    capital_to_use,

                "entry_cost":
                    entry_cost,

                "days_held":
                    0,

                "buy_probability":
                    float(
                        candidate[
                            "Buy_Probability"
                        ]
                    ),

                "expected_return":
                    float(
                        candidate[
                            "Expected_Return_20D"
                        ]
                    ),

                "technical_score":
                    float(
                        candidate[
                            "Technical_Score"
                        ]
                    ),

                "market_regime":
                    regime,

                "rank_score":
                    rank_score,

                "last_price":
                    entry_price
            }


            available_exposure -= (
                capital_to_use
            )


            selected += 1


        # ====================================================
        # DAILY EQUITY
        # ====================================================

        portfolio_value = cash


        for symbol, position in positions.items():

            price_row = get_price_row(
                symbol,
                current_date
            )


            if price_row is not None:

                price = float(
                    price_row["Close"]
                )

                position["last_price"] = (
                    price
                )

            else:

                price = position[
                    "last_price"
                ]


            portfolio_value += (
                position["shares"]
                * price
            )


        equity_history.append({

            "Date":
                current_date,

            "Portfolio_Value":
                portfolio_value,

            "Cash":
                cash,

            "Open_Positions":
                len(positions)
        })


    # ========================================================
    # CLOSE REMAINING
    # ========================================================

    if positions:

        final_date = pd.Timestamp(
            dates[-1]
        )


        for symbol, position in list(
            positions.items()
        ):

            price_row = get_price_row(
                symbol,
                final_date
            )


            if price_row is not None:

                exit_price = float(
                    price_row["Close"]
                )

            else:

                exit_price = position[
                    "last_price"
                ]


            exit_price *= (
                1 - SLIPPAGE
            )


            shares = position[
                "shares"
            ]


            gross_value = (
                shares
                * exit_price
            )


            exit_cost = (
                gross_value
                * TRANSACTION_COST
            )


            cash += (
                gross_value
                - exit_cost
            )


            gross_pnl = (
                shares
                * (
                    exit_price
                    - position[
                        "entry_price"
                    ]
                )
            )


            net_pnl = (
                gross_pnl
                - position[
                    "entry_cost"
                ]
                - exit_cost
            )


            trade_return = (
                net_pnl
                / position[
                    "capital_used"
                ]
            )


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

                "Capital_Used":
                    position[
                        "capital_used"
                    ],

                "Net_PnL":
                    net_pnl,

                "Return":
                    trade_return,

                "Days_Held":
                    position[
                        "days_held"
                    ],

                "Exit_Reason":
                    "END",

                "Buy_Probability":
                    position[
                        "buy_probability"
                    ],

                "Expected_Return_20D":
                    position[
                        "expected_return"
                    ],

                "Technical_Score":
                    position[
                        "technical_score"
                    ],

                "Market_Regime":
                    position[
                        "market_regime"
                    ],

                "Rank_Score":
                    position[
                        "rank_score"
                    ]
            })


            del positions[symbol]


    # ========================================================
    # METRICS
    # ========================================================

    equity = pd.DataFrame(
        equity_history
    )

    trades_df = pd.DataFrame(
        trades
    )


    if equity.empty:

        return {}


    equity["Peak"] = (
        equity[
            "Portfolio_Value"
        ]
        .cummax()
    )


    equity["Drawdown"] = (
        equity[
            "Portfolio_Value"
        ]
        / equity["Peak"]
        - 1
    )


    final_capital = (
        cash
    )


    total_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    )


    start_date = pd.Timestamp(
        equity["Date"].iloc[0]
    )

    end_date = pd.Timestamp(
        equity["Date"].iloc[-1]
    )


    days = (
        end_date
        - start_date
    ).days


    years = (
        days / 365.25
    )


    if years > 0:

        cagr = (
            final_capital
            / INITIAL_CAPITAL
        ) ** (
            1 / years
        ) - 1

    else:

        cagr = np.nan


    max_drawdown = float(
        equity[
            "Drawdown"
        ].min()
    )


    daily_returns = (
        equity[
            "Portfolio_Value"
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
        ) * np.sqrt(252)

    else:

        sharpe = 0.0


    if not trades_df.empty:

        win_rate = (
            trades_df[
                "Net_PnL"
            ] > 0
        ).mean()


        average_trade = (
            trades_df[
                "Return"
            ].mean()
        )

    else:

        win_rate = np.nan

        average_trade = np.nan


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nFinal Capital: "
        f"₹{final_capital:,.2f}"
    )

    print(
        f"Return: "
        f"{total_return:.2%}"
    )

    print(
        f"CAGR: "
        f"{cagr:.2%}"
    )

    print(
        f"Max Drawdown: "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Sharpe: "
        f"{sharpe:.2f}"
    )

    print(
        f"Trades: "
        f"{len(trades_df)}"
    )

    print(
        f"Win Rate: "
        f"{win_rate:.2%}"
    )

    print(
        f"Average Trade: "
        f"{average_trade:.2%}"
    )


    # ========================================================
    # REGIME PERFORMANCE
    # ========================================================

    if not trades_df.empty:

        regime_results = (

            trades_df
            .groupby(
                "Market_Regime"
            )
            .agg(

                Trades=(
                    "Net_PnL",
                    "count"
                ),

                Total_PnL=(
                    "Net_PnL",
                    "sum"
                ),

                Average_Return=(
                    "Return",
                    "mean"
                ),

                Win_Rate=(
                    "Net_PnL",
                    lambda x:
                    (x > 0).mean()
                )
            )

            .reset_index()
        )


        print(
            "\nRegime performance:"
        )

        print(
            regime_results.to_string(
                index=False
            )
        )

    else:

        regime_results = pd.DataFrame()


    return {

        "equity":
            equity,

        "trades":
            trades_df,

        "regime":
            regime_results,

        "metrics": {

            "Final_Capital":
                final_capital,

            "Return":
                total_return,

            "CAGR":
                cagr,

            "Max_Drawdown":
                max_drawdown,

            "Sharpe":
                sharpe,

            "Trades":
                len(trades_df),

            "Win_Rate":
                win_rate,

            "Average_Trade":
                average_trade
        }
    }


# ============================================================
# RUN DEVELOPMENT
# ============================================================

development_result = run_backtest(
    development,
    "Development"
)


# ============================================================
# RUN VALIDATION
# ============================================================

validation_result = run_backtest(
    validation,
    "Out-of-Sample Validation"
)


# ============================================================
# SAVE DEVELOPMENT
# ============================================================

if development_result:

    development_result[
        "equity"
    ].to_csv(
        OUTPUT_DIR
        / "development_equity.csv",
        index=False
    )


    development_result[
        "trades"
    ].to_csv(
        OUTPUT_DIR
        / "development_trades.csv",
        index=False
    )


    development_result[
        "regime"
    ].to_csv(
        OUTPUT_DIR
        / "development_regime.csv",
        index=False
    )


# ============================================================
# SAVE VALIDATION
# ============================================================

if validation_result:

    validation_result[
        "equity"
    ].to_csv(
        OUTPUT_DIR
        / "validation_equity.csv",
        index=False
    )


    validation_result[
        "trades"
    ].to_csv(
        OUTPUT_DIR
        / "validation_trades.csv",
        index=False
    )


    validation_result[
        "regime"
    ].to_csv(
        OUTPUT_DIR
        / "validation_regime.csv",
        index=False
    )


# ============================================================
# COMPARISON
# ============================================================

if (
    development_result
    and validation_result
):

    development_metrics = (
        development_result[
            "metrics"
        ]
    )

    validation_metrics = (
        validation_result[
            "metrics"
        ]
    )


    comparison = pd.DataFrame({

        "Metric": [

            "Final Capital",

            "Return",

            "CAGR",

            "Max Drawdown",

            "Sharpe",

            "Trades",

            "Win Rate",

            "Average Trade"
        ],

        "Development": [

            development_metrics[
                "Final_Capital"
            ],

            development_metrics[
                "Return"
            ],

            development_metrics[
                "CAGR"
            ],

            development_metrics[
                "Max_Drawdown"
            ],

            development_metrics[
                "Sharpe"
            ],

            development_metrics[
                "Trades"
            ],

            development_metrics[
                "Win_Rate"
            ],

            development_metrics[
                "Average_Trade"
            ]
        ],

        "Validation": [

            validation_metrics[
                "Final_Capital"
            ],

            validation_metrics[
                "Return"
            ],

            validation_metrics[
                "CAGR"
            ],

            validation_metrics[
                "Max_Drawdown"
            ],

            validation_metrics[
                "Sharpe"
            ],

            validation_metrics[
                "Trades"
            ],

            validation_metrics[
                "Win_Rate"
            ],

            validation_metrics[
                "Average_Trade"
            ]
        ]
    })


    comparison.to_csv(
        OUTPUT_DIR
        / "development_vs_validation.csv",
        index=False
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEVELOPMENT vs VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "V5 VALIDATION COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_DIR
)