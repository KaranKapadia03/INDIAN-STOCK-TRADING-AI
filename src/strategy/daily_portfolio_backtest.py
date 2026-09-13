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
    / "strategy_v4_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CAPITAL / EXECUTION
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
# V4 REGIME RULES
# ============================================================

# Bearish:
# The previous V3 test showed the strongest performance here.

BEARISH_MIN_BUY_PROB = 0.55
BEARISH_MIN_TECHNICAL = 0
BEARISH_MIN_EXPECTED_RETURN = 0.0


# Neutral:
# More selective because V3 lost money here.

NEUTRAL_MIN_BUY_PROB = 0.65
NEUTRAL_MIN_TECHNICAL = 1
NEUTRAL_MIN_EXPECTED_RETURN = 2.0


# Bullish:
# Most selective because V3 performed poorly here.

BULLISH_MIN_BUY_PROB = 0.75
BULLISH_MIN_TECHNICAL = 2
BULLISH_MIN_EXPECTED_RETURN = 4.0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("STRATEGY V4 - REGIME ADAPTIVE BACKTEST")
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
).reset_index(drop=True)


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

missing_columns = [
    column
    for column in required_columns
    if column not in predictions.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing_columns)
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
    ).reset_index(drop=True)

    return df


market_data = {}

symbols = sorted(
    predictions["Symbol"]
    .dropna()
    .unique()
)


for symbol in symbols:

    data = load_market_data(
        symbol
    )

    if data is not None:

        market_data[symbol] = data


print(
    f"Market datasets loaded: "
    f"{len(market_data)}/{len(symbols)}"
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
# V4 SIGNAL FUNCTION
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

        if buy_probability < BEARISH_MIN_BUY_PROB:

            return False

        if technical_score < BEARISH_MIN_TECHNICAL:

            return False

        if expected_return < BEARISH_MIN_EXPECTED_RETURN:

            return False

        return True


    # --------------------------------------------------------
    # BULLISH
    # --------------------------------------------------------

    if regime == "BULLISH":

        if buy_probability < BULLISH_MIN_BUY_PROB:

            return False

        if technical_score < BULLISH_MIN_TECHNICAL:

            return False

        if expected_return < BULLISH_MIN_EXPECTED_RETURN:

            return False

        return True


    # --------------------------------------------------------
    # NEUTRAL
    # --------------------------------------------------------

    if regime == "NEUTRAL":

        if buy_probability < NEUTRAL_MIN_BUY_PROB:

            return False

        if technical_score < NEUTRAL_MIN_TECHNICAL:

            return False

        if expected_return < NEUTRAL_MIN_EXPECTED_RETURN:

            return False

        return True


    return False


# ============================================================
# V4 SIGNALS
# ============================================================

predictions["V4_Candidate"] = predictions.apply(
    qualifies_for_v4,
    axis=1
)


predictions["V4_Signal"] = np.where(
    predictions["V4_Candidate"],
    "BUY",
    "HOLD"
)


print("\nV4 signal distribution:")

print(
    predictions[
        "V4_Signal"
    ]
    .value_counts()
    .to_string()
)


print("\nCandidates by regime:")

print(
    predictions[
        predictions["V4_Candidate"]
    ]
    ["Market_Regime"]
    .value_counts()
    .to_string()
)


# ============================================================
# RANKING
# ============================================================

def calculate_v4_rank(row):

    buy_probability = float(
        row["Buy_Probability"]
    )

    sell_probability = float(
        row["Sell_Probability"]
    )

    expected_return = float(
        row["Expected_Return_20D"]
    )

    technical_score = float(
        row["Technical_Score"]
    )

    regime = str(
        row["Market_Regime"]
    ).upper()


    ml_score = (
        buy_probability
        - sell_probability
    )


    expected_score = np.clip(
        expected_return / 10.0,
        -1.0,
        1.0
    )


    technical_score_normalized = np.clip(
        technical_score / 4.0,
        -1.0,
        1.0
    )


    # Regime is an adjustment, not a standalone signal.

    if regime == "BEARISH":

        regime_score = 0.15

    elif regime == "BULLISH":

        regime_score = 0.05

    else:

        regime_score = 0.0


    score = (

        0.45 * ml_score

        + 0.30 * expected_score

        + 0.20 * technical_score_normalized

        + 0.05 * regime_score
    )


    return score


predictions["V4_Rank_Score"] = predictions.apply(
    calculate_v4_rank,
    axis=1
)


# ============================================================
# PORTFOLIO STATE
# ============================================================

cash = INITIAL_CAPITAL

positions = {}

equity_history = []

trade_log = []

candidate_log = []


# ============================================================
# BACKTEST DATES
# ============================================================

all_dates = sorted(
    predictions["Date"].unique()
)


print(
    f"\nBacktest start: "
    f"{pd.Timestamp(all_dates[0]).date()}"
)

print(
    f"Backtest end: "
    f"{pd.Timestamp(all_dates[-1]).date()}"
)


# ============================================================
# MAIN BACKTEST
# ============================================================

for current_date in all_dates:

    current_date = pd.Timestamp(
        current_date
    )


    # ========================================================
    # 1. MANAGE EXISTING POSITIONS
    # ========================================================

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


        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        if low <= stop_price:

            exit_price = (
                stop_price
                * (1 - SLIPPAGE)
            )

            exit_reason = "STOP"


        # ----------------------------------------------------
        # TAKE PROFIT
        # ----------------------------------------------------

        if (
            exit_reason is None
            and high >= target_price
        ):

            exit_price = (
                target_price
                * (1 - SLIPPAGE)
            )

            exit_reason = "TARGET"


        # ----------------------------------------------------
        # TIME EXIT
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # EXECUTE EXIT
        # ----------------------------------------------------

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


            net_value = (
                gross_value
                - exit_cost
            )


            cash += net_value


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


            trade_log.append({

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

                "Shares":
                    shares,

                "Capital_Used":
                    position[
                        "capital_used"
                    ],

                "Gross_PnL":
                    gross_pnl,

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

                "Sell_Probability":
                    position[
                        "sell_probability"
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

                "Final_Score":
                    position[
                        "final_score"
                    ],

                "V4_Rank_Score":
                    position[
                        "rank_score"
                    ]
            })


            del positions[symbol]


    # ========================================================
    # 2. PORTFOLIO VALUE
    # ========================================================

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

            position["last_price"] = price

        else:

            price = position[
                "last_price"
            ]


        portfolio_value += (
            position["shares"]
            * price
        )


    # ========================================================
    # 3. EXPOSURE
    # ========================================================

    current_exposure = 0.0


    for symbol, position in positions.items():

        current_exposure += (
            position["shares"]
            * position["last_price"]
        )


    max_allowed_exposure = (
        portfolio_value
        * MAX_EXPOSURE
    )


    available_exposure = max(
        0.0,
        max_allowed_exposure
        - current_exposure
    )


    # ========================================================
    # 4. DAILY CANDIDATES
    # ========================================================

    daily_predictions = predictions[
        predictions["Date"]
        == current_date
    ].copy()


    candidates = daily_predictions[
        daily_predictions["V4_Candidate"]
    ].copy()


    candidates = candidates.sort_values(
        "V4_Rank_Score",
        ascending=False
    )


    # ========================================================
    # 5. ENTER POSITIONS
    # ========================================================

    selected_count = 0


    for _, candidate in candidates.iterrows():

        if selected_count >= MAX_POSITIONS:

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


        # ----------------------------------------------------
        # POSITION SIZE
        # ----------------------------------------------------

        rank_score = float(
            candidate[
                "V4_Rank_Score"
            ]
        )


        regime = str(
            candidate[
                "Market_Regime"
            ]
        ).upper()


        # Bearish trades get slightly more weight because
        # this regime performed best in V3.

        if regime == "BEARISH":

            if rank_score >= 0.30:

                position_fraction = 0.20

            else:

                position_fraction = 0.15


        elif regime == "BULLISH":

            position_fraction = 0.10


        else:

            if rank_score >= 0.30:

                position_fraction = 0.15

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


        # ----------------------------------------------------
        # EXECUTE ENTRY
        # ----------------------------------------------------

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

            "sell_probability":
                float(
                    candidate[
                        "Sell_Probability"
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

            "final_score":
                float(
                    candidate[
                        "Final_Score"
                    ]
                ),

            "rank_score":
                rank_score,

            "last_price":
                entry_price
        }


        available_exposure -= (
            capital_to_use
        )


        selected_count += 1


        # ----------------------------------------------------
        # LOG CANDIDATE
        # ----------------------------------------------------

        candidate_log.append({

            "Signal_Date":
                current_date,

            "Symbol":
                symbol,

            "Signal":
                "BUY",

            "Buy_Probability":
                candidate[
                    "Buy_Probability"
                ],

            "Sell_Probability":
                candidate[
                    "Sell_Probability"
                ],

            "Expected_Return_20D":
                candidate[
                    "Expected_Return_20D"
                ],

            "Technical_Score":
                candidate[
                    "Technical_Score"
                ],

            "Market_Regime":
                candidate[
                    "Market_Regime"
                ],

            "Final_Score":
                candidate[
                    "Final_Score"
                ],

            "V4_Rank_Score":
                rank_score,

            "Entry_Date":
                entry_date,

            "Entry_Price":
                entry_price,

            "Position_Fraction":
                position_fraction
        })


    # ========================================================
    # 6. DAILY EQUITY
    # ========================================================

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

            position["last_price"] = price

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

        "Cash":
            cash,

        "Portfolio_Value":
            portfolio_value,

        "Open_Positions":
            len(positions)
    })


# ============================================================
# CLOSE REMAINING POSITIONS
# ============================================================

if positions:

    final_date = pd.Timestamp(
        all_dates[-1]
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


        net_value = (
            gross_value
            - exit_cost
        )


        cash += net_value


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


        trade_log.append({

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
                shares,

            "Capital_Used":
                position[
                    "capital_used"
                ],

            "Gross_PnL":
                gross_pnl,

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

            "Sell_Probability":
                position[
                    "sell_probability"
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

            "Final_Score":
                position[
                    "final_score"
                ],

            "V4_Rank_Score":
                position[
                    "rank_score"
                ]
        })


        del positions[symbol]


# ============================================================
# EQUITY DATAFRAME
# ============================================================

equity = pd.DataFrame(
    equity_history
)


if not equity.empty:

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


# ============================================================
# TRADE DATAFRAME
# ============================================================

trades = pd.DataFrame(
    trade_log
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

final_capital = float(
    cash
)


total_return = (
    final_capital
    / INITIAL_CAPITAL
    - 1
)


if len(equity) > 1:

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

else:

    cagr = np.nan


if not equity.empty:

    max_drawdown = float(
        equity[
            "Drawdown"
        ].min()
    )

else:

    max_drawdown = np.nan


# ============================================================
# SHARPE
# ============================================================

if len(equity) > 1:

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

else:

    sharpe = np.nan


# ============================================================
# TRADE METRICS
# ============================================================

if not trades.empty:

    win_rate = float(
        (
            trades[
                "Net_PnL"
            ] > 0
        ).mean()
    )


    average_trade = float(
        trades[
            "Return"
        ].mean()
    )


    best_trade = float(
        trades[
            "Return"
        ].max()
    )


    worst_trade = float(
        trades[
            "Return"
        ].min()
    )

else:

    win_rate = np.nan

    average_trade = np.nan

    best_trade = np.nan

    worst_trade = np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STRATEGY V4 BACKTEST RESULTS"
)

print(
    "=" * 70
)


print(
    f"\nInitial Capital: "
    f"₹{INITIAL_CAPITAL:,.2f}"
)

print(
    f"Final Capital: "
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
    f"{len(trades)}"
)

print(
    f"Win Rate: "
    f"{win_rate:.2%}"
)

print(
    f"Average Trade: "
    f"{average_trade:.2%}"
)

print(
    f"Best Trade: "
    f"{best_trade:.2%}"
)

print(
    f"Worst Trade: "
    f"{worst_trade:.2%}"
)


# ============================================================
# EXIT REASONS
# ============================================================

if not trades.empty:

    print(
        "\nExit reasons:"
    )

    print(
        trades[
            "Exit_Reason"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# STOCK PERFORMANCE
# ============================================================

if not trades.empty:

    stock_results = (

        trades
        .groupby("Symbol")
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

        .sort_values(
            "Total_PnL",
            ascending=False
        )
    )


    print(
        "\nStock performance:"
    )

    print(
        stock_results.to_string(
            index=False
        )
    )

else:

    stock_results = pd.DataFrame()


# ============================================================
# REGIME PERFORMANCE
# ============================================================

if not trades.empty:

    regime_results = (

        trades
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

        .sort_values(
            "Total_PnL",
            ascending=False
        )
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


# ============================================================
# SAVE FILES
# ============================================================

equity.to_csv(
    OUTPUT_DIR
    / "daily_equity.csv",
    index=False
)


trades.to_csv(
    OUTPUT_DIR
    / "trades.csv",
    index=False
)


pd.DataFrame(
    candidate_log
).to_csv(
    OUTPUT_DIR
    / "trade_candidates.csv",
    index=False
)


summary = pd.DataFrame({

    "Metric": [

        "Initial Capital",

        "Final Capital",

        "Return",

        "CAGR",

        "Max Drawdown",

        "Sharpe",

        "Trades",

        "Win Rate",

        "Average Trade",

        "Best Trade",

        "Worst Trade"
    ],

    "Value": [

        INITIAL_CAPITAL,

        final_capital,

        total_return,

        cagr,

        max_drawdown,

        sharpe,

        len(trades),

        win_rate,

        average_trade,

        best_trade,

        worst_trade
    ]
})


summary.to_csv(
    OUTPUT_DIR
    / "summary.csv",
    index=False
)


stock_results.to_csv(
    OUTPUT_DIR
    / "stock_performance.csv",
    index=False
)


regime_results.to_csv(
    OUTPUT_DIR
    / "regime_performance.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FILES SAVED"
)

print("=" * 70)

print(
    f"\n{OUTPUT_DIR}"
)

print(
    "\nStrategy V4 completed successfully."
)