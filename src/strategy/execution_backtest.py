
import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)

INPUT_FILE = os.path.join(
    MODEL_DIR,
    "true_walk_forward",
    "walk_forward_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    "execution_backtest"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

POSITION_SIZE = 0.20

MAX_TOTAL_EXPOSURE = 0.80

HOLD_DAYS = 20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

# Risk management
STOP_LOSS_PERCENT = 0.06

TAKE_PROFIT_PERCENT = 0.12


# ============================================================
# LOAD ORIGINAL OHLC DATA
# ============================================================

def load_ohlc(symbol):

    filename = (
        symbol.replace(
            ".",
            "_"
        )
        + ".csv"
    )

    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"OHLC data missing:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Yahoo Finance data may contain timezone-aware dates.
    #
    # Predictions use timezone-naive dates.
    #
    # Convert both to the same timezone-naive format.
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True
    )

    df["Date"] = (
        df["Date"]
        .dt.tz_localize(None)
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Prediction file missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    # Predictions are timezone-naive.
    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Explicitly remove timezone if one somehow exists.
    if hasattr(
        df["date"].dt,
        "tz"
    ):

        if df["date"].dt.tz is not None:

            df["date"] = (
                df["date"]
                .dt.tz_localize(None)
            )

    return df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


# ============================================================
# BUILD EXECUTION DATA
# ============================================================

def prepare_execution_data(
    predictions
):

    all_rows = []

    for symbol in predictions[
        "symbol"
    ].unique():

        try:

            ohlc = load_ohlc(
                symbol
            )

        except FileNotFoundError:

            print(
                f"Missing OHLC: {symbol}"
            )

            continue

        symbol_predictions = (
            predictions[
                predictions[
                    "symbol"
                ]
                ==
                symbol
            ]
            .copy()
        )

        symbol_predictions["date"] = (
            pd.to_datetime(
                symbol_predictions[
                    "date"
                ]
            )
            .dt.tz_localize(None)
        )

        symbol_predictions = (
            symbol_predictions
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        ohlc = (
            ohlc
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Find the NEXT trading day.
        #
        # Signal is generated after today's close.
        # Entry therefore happens on the next trading day.
        # ----------------------------------------------------

        merged = pd.merge_asof(

            symbol_predictions,

            ohlc,

            left_on="date",

            right_on="Date",

            direction="forward",

            allow_exact_matches=False
        )

        # ----------------------------------------------------
        # Remove rows where a next trading day wasn't found.
        # ----------------------------------------------------

        merged = merged[
            merged["Date"].notna()
        ].copy()

        # ----------------------------------------------------
        # Rename execution information.
        # ----------------------------------------------------

        merged["entry_date"] = (
            merged["Date"]
        )

        merged["entry_open"] = (
            merged["Open"]
        )

        merged["entry_high"] = (
            merged["High"]
        )

        merged["entry_low"] = (
            merged["Low"]
        )

        merged["entry_close"] = (
            merged["Close"]
        )

        all_rows.append(
            merged
        )

    if not all_rows:

        return pd.DataFrame()

    return pd.concat(
        all_rows,
        ignore_index=True
    )


# ============================================================
# FIND FUTURE OHLC
# ============================================================

def get_future_ohlc(
    symbol,
    entry_date,
    days
):

    ohlc = load_ohlc(
        symbol
    )

    future = ohlc[
        ohlc["Date"]
        >=
        entry_date
    ].copy()

    if future.empty:

        return None

    future = (
        future
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    # Entry day is day 0.
    # We inspect the following trading sessions.
    future = future.iloc[
        : days + 1
    ]

    return future


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    row
):

    symbol = row[
        "symbol"
    ]

    entry_date = row[
        "entry_date"
    ]

    entry_price = float(
        row[
            "entry_open"
        ]
    )

    if entry_price <= 0:

        return None

    future = get_future_ohlc(
        symbol,
        entry_date,
        HOLD_DAYS
    )

    if future is None:

        return None

    if len(future) < 2:

        return None

    # --------------------------------------------------------
    # Stop and target
    # --------------------------------------------------------

    stop_price = (
        entry_price
        *
        (
            1
            -
            STOP_LOSS_PERCENT
        )
    )

    target_price = (
        entry_price
        *
        (
            1
            +
            TAKE_PROFIT_PERCENT
        )
    )

    exit_price = None

    exit_date = None

    exit_reason = None

    # --------------------------------------------------------
    # Check future trading sessions
    # --------------------------------------------------------

    for i in range(
        1,
        len(future)
    ):

        day = future.iloc[i]

        low = float(
            day["Low"]
        )

        high = float(
            day["High"]
        )

        # ----------------------------------------------------
        # Conservative assumption:
        #
        # If both stop and target are hit during the same
        # candle, assume the stop was hit first.
        # ----------------------------------------------------

        if low <= stop_price:

            exit_price = (
                stop_price
            )

            exit_date = (
                day["Date"]
            )

            exit_reason = "STOP"

            break

        if high >= target_price:

            exit_price = (
                target_price
            )

            exit_date = (
                day["Date"]
            )

            exit_reason = "TARGET"

            break

    # --------------------------------------------------------
    # If neither stop nor target was reached,
    # exit at the final day's closing price.
    # --------------------------------------------------------

    if exit_price is None:

        last_day = future.iloc[
            -1
        ]

        exit_price = float(
            last_day["Close"]
        )

        exit_date = (
            last_day["Date"]
        )

        exit_reason = "TIME"

    # --------------------------------------------------------
    # Trading costs
    # --------------------------------------------------------

    gross_return = (
        exit_price
        /
        entry_price
        - 1
    )

    net_return = (
        gross_return
        -
        TRANSACTION_COST
        -
        SLIPPAGE
    )

    return {

        "symbol":
            symbol,

        "signal_date":
            row["date"],

        "entry_date":
            entry_date,

        "exit_date":
            exit_date,

        "entry_price":
            entry_price,

        "exit_price":
            exit_price,

        "gross_return":
            gross_return * 100,

        "net_return":
            net_return * 100,

        "exit_reason":
            exit_reason,

        "buy_probability":
            row[
                "buy_probability"
            ],

        "sell_probability":
            row[
                "sell_probability"
            ],

        "expected_return":
            row[
                "expected_return_20d"
            ],

        "final_score":
            row[
                "final_score"
            ]
    }


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def run_portfolio(
    execution_data
):

    execution_data = (
        execution_data
        .copy()
    )

    execution_data["entry_date"] = (
        pd.to_datetime(
            execution_data[
                "entry_date"
            ]
        )
    )

    dates = sorted(
        execution_data[
            "entry_date"
        ]
        .dropna()
        .unique()
    )

    cash = INITIAL_CAPITAL

    open_positions = []

    trades = []

    equity_history = []

    # ========================================================
    # PROCESS EACH TRADING DATE
    # ========================================================

    for current_date in dates:

        # ====================================================
        # CLOSE EXISTING POSITIONS
        # ====================================================

        remaining_positions = []

        for position in open_positions:

            if position[
                "exit_date"
            ] <= current_date:

                capital = position[
                    "capital"
                ]

                net_return = (
                    position[
                        "net_return"
                    ]
                    /
                    100
                )

                pnl = (
                    capital
                    *
                    net_return
                )

                cash += (
                    capital
                    +
                    pnl
                )

                position[
                    "pnl"
                ] = pnl

                trades.append(
                    position
                )

            else:

                remaining_positions.append(
                    position
                )

        open_positions = (
            remaining_positions
        )

        # ====================================================
        # CURRENT PORTFOLIO VALUE
        # ====================================================

        invested = sum(
            position[
                "capital"
            ]
            for position
            in open_positions
        )

        portfolio_value = (
            cash
            +
            invested
        )

        exposure = (
            invested
            /
            max(
                portfolio_value,
                1
            )
        )

        # ====================================================
        # BUY SIGNALS FOR TODAY'S OPEN
        # ====================================================

        candidates = (
            execution_data[
                execution_data[
                    "entry_date"
                ]
                ==
                current_date
            ]
        )

        candidates = candidates[
            candidates[
                "signal"
            ]
            ==
            "BUY"
        ].copy()

        candidates = (
            candidates
            .sort_values(
                "final_score",
                ascending=False
            )
        )

        # ====================================================
        # OPEN NEW POSITIONS
        # ====================================================

        for _, row in candidates.iterrows():

            symbol = row[
                "symbol"
            ]

            # ------------------------------------------------
            # Don't hold the same stock twice.
            # ------------------------------------------------

            already_holding = any(

                position[
                    "symbol"
                ]
                ==
                symbol

                for position
                in open_positions
            )

            if already_holding:

                continue

            # ------------------------------------------------
            # Maximum portfolio exposure
            # ------------------------------------------------

            if exposure >= MAX_TOTAL_EXPOSURE:

                break

            available = (
                MAX_TOTAL_EXPOSURE
                -
                exposure
            )

            allocation = min(
                POSITION_SIZE,
                available
            )

            if allocation <= 0:

                break

            # ------------------------------------------------
            # Position capital
            # ------------------------------------------------

            capital = (
                portfolio_value
                *
                allocation
            )

            if capital > cash:

                continue

            # ------------------------------------------------
            # Simulate trade
            # ------------------------------------------------

            trade = simulate_trade(
                row
            )

            if trade is None:

                continue

            trade[
                "capital"
            ] = capital

            trade[
                "pnl"
            ] = 0

            open_positions.append(
                trade
            )

            cash -= capital

            invested += capital

            exposure = (
                invested
                /
                max(
                    portfolio_value,
                    1
                )
            )

        # ====================================================
        # RECORD EQUITY
        # ====================================================

        invested = sum(
            position[
                "capital"
            ]
            for position
            in open_positions
        )

        portfolio_value = (
            cash
            +
            invested
        )

        equity_history.append({

            "date":
                current_date,

            "portfolio_value":
                portfolio_value,

            "cash":
                cash,

            "invested":
                invested,

            "positions":
                len(
                    open_positions
                )
        })

    # ========================================================
    # CLOSE REMAINING POSITIONS
    # ========================================================

    for position in open_positions:

        capital = position[
            "capital"
        ]

        net_return = (
            position[
                "net_return"
            ]
            /
            100
        )

        pnl = (
            capital
            *
            net_return
        )

        cash += (
            capital
            +
            pnl
        )

        position[
            "pnl"
        ] = pnl

        trades.append(
            position
        )

    equity = pd.DataFrame(
        equity_history
    )

    trades = pd.DataFrame(
        trades
    )

    return (
        equity,
        trades
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    equity,
    trades
):

    if equity.empty:

        return {}

    equity = (
        equity
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    equity[
        "daily_return"
    ] = (
        equity[
            "portfolio_value"
        ]
        .pct_change()
    )

    running_max = (
        equity[
            "portfolio_value"
        ]
        .cummax()
    )

    drawdown = (
        equity[
            "portfolio_value"
        ]
        /
        running_max
        - 1
    )

    initial = float(
        equity[
            "portfolio_value"
        ].iloc[0]
    )

    final = float(
        equity[
            "portfolio_value"
        ].iloc[-1]
    )

    total_return = (
        final
        /
        initial
        - 1
    ) * 100

    max_drawdown = (
        drawdown.min()
        *
        100
    )

    daily = (
        equity[
            "daily_return"
        ]
        .dropna()
    )

    if (
        len(daily) > 1
        and
        daily.std() > 0
    ):

        sharpe = (
            daily.mean()
            /
            daily.std()
        ) * np.sqrt(252)

    else:

        sharpe = 0.0

    if trades.empty:

        win_rate = 0.0

        avg_trade = 0.0

    else:

        win_rate = (
            trades[
                "net_return"
            ]
            > 0
        ).mean() * 100

        avg_trade = (
            trades[
                "net_return"
            ].mean()
        )

    return {

        "initial":
            initial,

        "final":
            final,

        "return":
            total_return,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe,

        "trades":
            len(trades),

        "win_rate":
            win_rate,

        "avg_trade":
            avg_trade
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "INDIAN STOCK TRADING AI"
    )

    print(
        "REALISTIC NEXT-DAY EXECUTION BACKTEST"
    )

    print(
        "=" * 110
    )

    # --------------------------------------------------------
    # Load walk-forward predictions
    # --------------------------------------------------------

    predictions = (
        load_predictions()
    )

    print(
        f"\nPredictions loaded: "
        f"{len(predictions)}"
    )

    # --------------------------------------------------------
    # Match every signal with the NEXT trading day's OHLC.
    # --------------------------------------------------------

    execution_data = (
        prepare_execution_data(
            predictions
        )
    )

    print(
        f"Execution rows created: "
        f"{len(execution_data)}"
    )

    # --------------------------------------------------------
    # Only BUY signals are actually entered.
    # --------------------------------------------------------

    execution_data = (
        execution_data[
            execution_data[
                "signal"
            ]
            ==
            "BUY"
        ]
        .copy()
    )

    print(
        f"BUY signals available: "
        f"{len(execution_data)}"
    )

    if execution_data.empty:

        print(
            "\nNo BUY signals available."
        )

        return

    # --------------------------------------------------------
    # Run portfolio
    # --------------------------------------------------------

    (
        equity,
        trades
    ) = run_portfolio(
        execution_data
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(
        equity,
        trades
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    equity_file = os.path.join(
        OUTPUT_DIR,
        "equity.csv"
    )

    trades_file = os.path.join(
        OUTPUT_DIR,
        "trades.csv"
    )

    execution_file = os.path.join(
        OUTPUT_DIR,
        "execution_data.csv"
    )

    equity.to_csv(
        equity_file,
        index=False
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    execution_data.to_csv(
        execution_file,
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")

    print(
        "=" * 90
    )

    print(
        "REALISTIC EXECUTION RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"Initial capital : "
        f"₹{stats['initial']:,.2f}"
    )

    print(
        f"Final capital   : "
        f"₹{stats['final']:,.2f}"
    )

    print(
        f"Return          : "
        f"{stats['return']:+.2f}%"
    )

    print(
        f"Max drawdown    : "
        f"{stats['max_drawdown']:+.2f}%"
    )

    print(
        f"Sharpe          : "
        f"{stats['sharpe']:.2f}"
    )

    print(
        f"Trades          : "
        f"{stats['trades']}"
    )

    print(
        f"Win rate        : "
        f"{stats['win_rate']:.2f}%"
    )

    print(
        f"Avg trade       : "
        f"{stats['avg_trade']:+.2f}%"
    )

    # ========================================================
    # EXIT REASONS
    # ========================================================

    if not trades.empty:

        print("\n")

        print(
            "=" * 70
        )

        print(
            "EXIT REASONS"
        )

        print(
            "=" * 70
        )

        print(
            trades[
                "exit_reason"
            ]
            .value_counts()
        )

        print("\n")

        print(
            "BEST TRADE : "
            f"{trades['net_return'].max():+.2f}%"
        )

        print(
            "WORST TRADE: "
            f"{trades['net_return'].min():+.2f}%"
        )

    # ========================================================
    # FILES
    # ========================================================

    print("\n")

    print(
        "=" * 110
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 110
    )

    print(
        equity_file
    )

    print(
        trades_file
    )

    print(
        execution_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

