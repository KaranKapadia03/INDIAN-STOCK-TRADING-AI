
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

OUTPUT_DIR = Path(
    "data/models/v29_realistic_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

MIN_TRAIN_DAYS = 500

RETRAIN_DAYS = 20

TOP_N = 3

INITIAL_CAPITAL = 100_000.0

MAX_EXPOSURE = 0.80

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

STOP_LOSS = 0.06

TAKE_PROFIT = 0.12

MAX_HOLD_DAYS = 20

RANDOM_STATE = 42


FEATURES = [
    "Volatility_20D",
    "Volume_Ratio",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
]


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
# SYMBOL EXTRACTION
# ============================================================

def extract_symbol(path):

    name = path.stem

    name = name.replace(
        "_features",
        ""
    )

    if name.endswith("_NS"):

        name = (
            name[:-3]
            + ".NS"
        )

    return name


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_data():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    print(
        f"Feature files found: "
        f"{len(files)}"
    )

    datasets = []

    for file in files:

        symbol = extract_symbol(
            file
        )

        df = pd.read_csv(
            file
        )

        required = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "NIFTY_Return_20D",
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            print(
                f"Skipping {symbol}: "
                f"missing {missing}"
            )

            continue

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        df["Symbol"] = symbol

        datasets.append(df)

    if not datasets:

        raise RuntimeError(
            "No usable stock datasets."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.loc[
        :,
        ~data.columns.duplicated()
    ]

    data = (
        data
        .sort_values(
            ["Date", "Symbol"]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(data):

    available = [
        feature
        for feature in FEATURES
        if feature in data.columns
    ]

    if len(available) != len(FEATURES):

        missing = [
            feature
            for feature in FEATURES
            if feature not in data.columns
        ]

        raise RuntimeError(
            f"Missing features: {missing}"
        )

    df = data.copy()

    # --------------------------------------------------------
    # Future 20D return
    # --------------------------------------------------------

    df[
        "Future_Return_20D"
    ] = (

        df
        .groupby("Symbol")["Close"]
        .shift(-HORIZON)

        / df["Close"]

        - 1

    ) * 100

    # --------------------------------------------------------
    # Fill feature values cross-sectionally
    # --------------------------------------------------------

    for feature in available:

        df[feature] = (

            df
            .groupby("Date")[feature]
            .transform(
                lambda x:
                x.fillna(
                    x.median()
                )
            )

        )

        median_value = (
            df[feature].median()
        )

        df[feature] = (
            df[feature]
            .fillna(median_value)
        )

    df = df.dropna(
        subset=[
            "Future_Return_20D"
        ] + available
    )

    return (
        df
        .sort_values(
            ["Date", "Symbol"]
        )
        .reset_index(drop=True),
        available
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return RandomForestRegressor(

        n_estimators=400,

        max_depth=8,

        min_samples_split=15,

        min_samples_leaf=5,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def generate_predictions(
    data,
    features
):

    dates = sorted(
        data["Date"].unique()
    )

    predictions = []

    rebalance_id = 0

    for i in range(
        MIN_TRAIN_DAYS,
        len(dates) - HORIZON,
        RETRAIN_DAYS,
    ):

        test_date = dates[i]

        # ----------------------------------------------------
        # Strict leakage prevention.
        #
        # The model cannot see any observation whose
        # 20-day future return would overlap the test date.
        # ----------------------------------------------------

        cutoff_date = dates[
            i - HORIZON
        ]

        train = data[
            data["Date"]
            <= cutoff_date
        ].copy()

        test = data[
            data["Date"]
            == test_date
        ].copy()

        if train.empty or test.empty:
            continue

        if (
            train["Date"].nunique()
            < MIN_TRAIN_DAYS
        ):
            continue

        model = create_model()

        model.fit(
            train[features],
            train[
                "Future_Return_20D"
            ],
        )

        test = test.copy()

        test[
            "Prediction"
        ] = model.predict(
            test[features]
        )

        test[
            "Rebalance_ID"
        ] = rebalance_id

        test[
            "Train_End"
        ] = cutoff_date

        predictions.append(
            test[
                [
                    "Date",
                    "Symbol",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Prediction",
                    "Rebalance_ID",
                    "Train_End",
                ]
            ]
        )

        rebalance_id += 1

    if not predictions:

        return pd.DataFrame()

    return (
        pd.concat(
            predictions,
            ignore_index=True
        )
        .sort_values(
            ["Date", "Prediction"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )


# ============================================================
# REALISTIC TRADE SIMULATION
# ============================================================

def simulate_portfolio(
    predictions,
    raw_data
):

    if predictions.empty:
        return None

    trading_dates = sorted(
        raw_data["Date"].unique()
    )

    # Only rebalance dates for which we
    # actually have predictions.
    rebalance_dates = sorted(
        predictions["Date"].unique()
    )

    capital = (
        INITIAL_CAPITAL
    )

    positions = []

    trades = []

    equity_curve = []

    previous_rebalance = None

    for rebalance_date in rebalance_dates:

        # ----------------------------------------------------
        # First mark any existing positions to the current
        # market price and close positions whose maximum
        # holding period has expired.
        # ----------------------------------------------------

        new_positions = []

        for position in positions:

            symbol = position[
                "Symbol"
            ]

            entry_date = position[
                "Entry_Date"
            ]

            held_days = (
                trading_dates.index(
                    rebalance_date
                )
                - trading_dates.index(
                    entry_date
                )
            )

            stock = raw_data[
                (
                    raw_data["Symbol"]
                    == symbol
                )
                &
                (
                    raw_data["Date"]
                    == rebalance_date
                )
            ]

            if stock.empty:

                new_positions.append(
                    position
                )

                continue

            row = stock.iloc[0]

            exit_reason = None
            exit_price = None

            # ------------------------------------------------
            # High/Low based stop and target.
            #
            # Conservative assumption:
            # if both stop and target are touched on the
            # same day, STOP is assumed to execute first.
            # ------------------------------------------------

            stop_price = position[
                "Entry_Price"
            ] * (
                1 - STOP_LOSS
            )

            target_price = position[
                "Entry_Price"
            ] * (
                1 + TAKE_PROFIT
            )

            if row["Low"] <= stop_price:

                exit_reason = "STOP"

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE)
                )

            elif row["High"] >= target_price:

                exit_reason = "TARGET"

                exit_price = (
                    target_price
                    * (1 - SLIPPAGE)
                )

            elif held_days >= MAX_HOLD_DAYS:

                exit_reason = "TIME"

                exit_price = (
                    row["Open"]
                    * (1 - SLIPPAGE)
                )

            if exit_reason is None:

                new_positions.append(
                    position
                )

                continue

            shares = position[
                "Shares"
            ]

            gross_value = (
                shares
                * exit_price
            )

            exit_cost = (
                gross_value
                * TRANSACTION_COST
            )

            proceeds = (
                gross_value
                - exit_cost
            )

            entry_value = position[
                "Entry_Value"
            ]

            pnl = (
                proceeds
                - entry_value
            )

            net_return = (
                pnl
                / entry_value
            )

            capital += proceeds

            trades.append({

                "Symbol":
                    symbol,

                "Entry_Date":
                    entry_date,

                "Exit_Date":
                    rebalance_date,

                "Entry_Price":
                    position[
                        "Entry_Price"
                    ],

                "Exit_Price":
                    exit_price,

                "Shares":
                    shares,

                "Prediction":
                    position[
                        "Prediction"
                    ],

                "Exit_Reason":
                    exit_reason,

                "Entry_Value":
                    entry_value,

                "Exit_Value":
                    proceeds,

                "PnL":
                    pnl,

                "Return":
                    net_return,

                "Hold_Days":
                    held_days,

            })

        positions = new_positions

        # ----------------------------------------------------
        # Rebalance
        # ----------------------------------------------------

        day_predictions = predictions[
            predictions["Date"]
            == rebalance_date
        ].copy()

        if day_predictions.empty:

            continue

        # Only buy the highest-ranked stocks.
        selected = (
            day_predictions
            .sort_values(
                "Prediction",
                ascending=False
            )
            .head(TOP_N)
        )

        # ----------------------------------------------------
        # Do not buy stocks already held.
        # ----------------------------------------------------

        held_symbols = {
            position["Symbol"]
            for position in positions
        }

        selected = selected[
            ~selected["Symbol"]
            .isin(held_symbols)
        ]

        if not selected.empty:

            available_capital = (
                capital
                * MAX_EXPOSURE
            )

            position_value = (
                available_capital
                / TOP_N
            )

            for _, prediction in (
                selected.iterrows()
            ):

                symbol = prediction[
                    "Symbol"
                ]

                stock = raw_data[
                    (
                        raw_data["Symbol"]
                        == symbol
                    )
                    &
                    (
                        raw_data["Date"]
                        == rebalance_date
                    )
                ]

                if stock.empty:
                    continue

                row = stock.iloc[0]

                # ------------------------------------------------
                # Enter at next trading day's OPEN.
                # ------------------------------------------------

                current_index = (
                    trading_dates.index(
                        rebalance_date
                    )
                )

                next_index = (
                    current_index + 1
                )

                if (
                    next_index
                    >= len(trading_dates)
                ):
                    continue

                entry_date = trading_dates[
                    next_index
                ]

                entry_row = raw_data[
                    (
                        raw_data["Symbol"]
                        == symbol
                    )
                    &
                    (
                        raw_data["Date"]
                        == entry_date
                    )
                ]

                if entry_row.empty:
                    continue

                entry_row = (
                    entry_row.iloc[0]
                )

                entry_price = (
                    entry_row["Open"]
                    * (1 + SLIPPAGE)
                )

                entry_cost = (
                    position_value
                    * TRANSACTION_COST
                )

                total_required = (
                    position_value
                    + entry_cost
                )

                if (
                    total_required
                    > capital
                ):
                    continue

                shares = (
                    position_value
                    / entry_price
                )

                capital -= (
                    total_required
                )

                positions.append({

                    "Symbol":
                        symbol,

                    "Entry_Date":
                        entry_date,

                    "Entry_Price":
                        entry_price,

                    "Shares":
                        shares,

                    "Entry_Value":
                        position_value,

                    "Prediction":
                        prediction[
                            "Prediction"
                        ],

                })

        # ----------------------------------------------------
        # Portfolio mark-to-market
        # ----------------------------------------------------

        equity = capital

        for position in positions:

            stock = raw_data[
                (
                    raw_data["Symbol"]
                    == position[
                        "Symbol"
                    ]
                )
                &
                (
                    raw_data["Date"]
                    == rebalance_date
                )
            ]

            if not stock.empty:

                close = (
                    stock.iloc[0]["Close"]
                )

                equity += (
                    position["Shares"]
                    * close
                )

            else:

                equity += (
                    position[
                        "Entry_Value"
                    ]
                )

        equity_curve.append({

            "Date":
                rebalance_date,

            "Equity":
                equity,

            "Cash":
                capital,

            "Positions":
                len(positions),

        })

        previous_rebalance = (
            rebalance_date
        )

    # ========================================================
    # FORCE CLOSE REMAINING POSITIONS
    # ========================================================

    if positions:

        final_date = (
            trading_dates[-1]
        )

        for position in positions:

            symbol = position[
                "Symbol"
            ]

            stock = raw_data[
                (
                    raw_data["Symbol"]
                    == symbol
                )
                &
                (
                    raw_data["Date"]
                    == final_date
                )
            ]

            if stock.empty:
                continue

            close_price = (
                stock.iloc[0]["Close"]
                * (1 - SLIPPAGE)
            )

            shares = position[
                "Shares"
            ]

            gross_value = (
                shares
                * close_price
            )

            exit_cost = (
                gross_value
                * TRANSACTION_COST
            )

            proceeds = (
                gross_value
                - exit_cost
            )

            pnl = (
                proceeds
                - position[
                    "Entry_Value"
                ]
            )

            net_return = (
                pnl
                / position[
                    "Entry_Value"
                ]
            )

            capital += proceeds

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
                    close_price,

                "Shares":
                    shares,

                "Prediction":
                    position[
                        "Prediction"
                    ],

                "Exit_Reason":
                    "END",

                "Entry_Value":
                    position[
                        "Entry_Value"
                    ],

                "Exit_Value":
                    proceeds,

                "PnL":
                    pnl,

                "Return":
                    net_return,

                "Hold_Days":
                    (
                        trading_dates.index(
                            final_date
                        )
                        -
                        trading_dates.index(
                            position[
                                "Entry_Date"
                            ]
                        )
                    ),

            })

    if not equity_curve:

        return None

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    # ========================================================
    # METRICS
    # ========================================================

    equity_df[
        "Peak"
    ] = (
        equity_df["Equity"]
        .cummax()
    )

    equity_df[
        "Drawdown"
    ] = (

        equity_df["Equity"]
        / equity_df["Peak"]
        - 1

    )

    final_equity = capital

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    start_date = pd.Timestamp(
        equity_df["Date"].iloc[0]
    )

    end_date = pd.Timestamp(
        equity_df["Date"].iloc[-1]
    )

    years = max(
        (
            end_date
            - start_date
        ).days
        / 365.25,
        0.1
    )

    cagr = (
        (
            final_equity
            / INITIAL_CAPITAL
        )
        ** (1 / years)
        - 1
    )

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

        sharpe = np.nan

    if len(trades_df):

        win_rate = (
            trades_df[
                "Return"
            ] > 0
        ).mean()

        avg_trade = (
            trades_df[
                "Return"
            ].mean()
        )

        gross_profit = (
            trades_df.loc[
                trades_df[
                    "Return"
                ] > 0,
                "PnL"
            ].sum()
        )

        gross_loss = abs(
            trades_df.loc[
                trades_df[
                    "Return"
                ] < 0,
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

        win_rate = np.nan

        avg_trade = np.nan

        profit_factor = np.nan

    return {

        "Final_Equity":
            final_equity,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            equity_df[
                "Drawdown"
            ].min(),

        "Sharpe":
            sharpe,

        "Win_Rate":
            win_rate,

        "Profit_Factor":
            profit_factor,

        "Avg_Trade":
            avg_trade,

        "Trades":
            len(trades_df),

        "Equity":
            equity_df,

        "Trades_Data":
            trades_df,

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
        "V29 - REALISTIC EXECUTION BACKTEST"
    )

    print("=" * 70)

    print(
        "\nLocked feature set:"
    )

    for feature in FEATURES:

        print(
            f"  - {feature}"
        )

    data = load_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    prepared, features = (
        prepare_data(data)
    )

    print(
        f"Usable rows: "
        f"{len(prepared):,}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Generate strict walk-forward predictions
    # --------------------------------------------------------

    predictions = generate_predictions(
        prepared,
        features
    )

    if predictions.empty:

        raise RuntimeError(
            "No predictions generated."
        )

    print(
        f"\nPredictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Rebalances: "
        f"{predictions['Date'].nunique()}"
    )

    print(
        f"Period: "
        f"{predictions['Date'].min().date()}"
        f" -> "
        f"{predictions['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions.to_csv(

        OUTPUT_DIR
        / "v29_predictions.csv",

        index=False

    )

    # --------------------------------------------------------
    # Run execution simulation
    # --------------------------------------------------------

    result = simulate_portfolio(
        predictions,
        data
    )

    if result is None:

        raise RuntimeError(
            "Backtest failed."
        )

    print("\n")
    print("=" * 70)

    print(
        "V29 REALISTIC BACKTEST RESULTS"
    )

    print("=" * 70)

    print(
        f"Final Equity: "
        f"₹{result['Final_Equity']:,.2f}"
    )

    print(
        f"Total Return: "
        f"{result['Total_Return'] * 100:.2f}%"
    )

    print(
        f"CAGR: "
        f"{result['CAGR'] * 100:.2f}%"
    )

    print(
        f"Max Drawdown: "
        f"{result['Max_Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe: "
        f"{result['Sharpe']:.2f}"
    )

    print(
        f"Win Rate: "
        f"{result['Win_Rate'] * 100:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{result['Profit_Factor']:.2f}"
    )

    print(
        f"Average Trade: "
        f"{result['Avg_Trade'] * 100:.3f}%"
    )

    print(
        f"Trades: "
        f"{result['Trades']}"
    )

    # --------------------------------------------------------
    # Exit analysis
    # --------------------------------------------------------

    trades = result[
        "Trades_Data"
    ]

    if not trades.empty:

        print("\n")
        print(
            "=" * 70
        )

        print(
            "EXIT ANALYSIS"
        )

        print(
            "=" * 70
        )

        exit_summary = (
            trades
            .groupby(
                "Exit_Reason"
            )
            .agg(

                Trades=(
                    "Return",
                    "count"
                ),

                Avg_Return=(
                    "Return",
                    "mean"
                ),

                Total_PnL=(
                    "PnL",
                    "sum"
                ),

                Win_Rate=(
                    "Return",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                ),

            )

            .reset_index()
        )

        exit_summary[
            "Avg_Return"
        ] *= 100

        exit_summary[
            "Win_Rate"
        ] *= 100

        print(
            exit_summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Stock attribution
    # --------------------------------------------------------

    if not trades.empty:

        print("\n")
        print(
            "=" * 70
        )

        print(
            "STOCK ATTRIBUTION"
        )

        print(
            "=" * 70
        )

        attribution = (
            trades
            .groupby("Symbol")
            .agg(

                Trades=(
                    "Return",
                    "count"
                ),

                Total_PnL=(
                    "PnL",
                    "sum"
                ),

                Avg_Return=(
                    "Return",
                    "mean"
                ),

                Win_Rate=(
                    "Return",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                ),

            )

            .sort_values(
                "Total_PnL",
                ascending=False
            )

            .reset_index()
        )

        attribution[
            "Avg_Return"
        ] *= 100

        attribution[
            "Win_Rate"
        ] *= 100

        print(
            attribution.to_string(
                index=False
            )
        )

        attribution.to_csv(

            OUTPUT_DIR
            / "v29_stock_attribution.csv",

            index=False

        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    result[
        "Equity"
    ].to_csv(

        OUTPUT_DIR
        / "v29_equity_curve.csv",

        index=False

    )

    result[
        "Trades_Data"
    ].to_csv(

        OUTPUT_DIR
        / "v29_trade_log.csv",

        index=False

    )

    summary = pd.DataFrame([{

        "Final_Equity":
            result[
                "Final_Equity"
            ],

        "Total_Return":
            result[
                "Total_Return"
            ],

        "CAGR":
            result[
                "CAGR"
            ],

        "Max_Drawdown":
            result[
                "Max_Drawdown"
            ],

        "Sharpe":
            result[
                "Sharpe"
            ],

        "Win_Rate":
            result[
                "Win_Rate"
            ],

        "Profit_Factor":
            result[
                "Profit_Factor"
            ],

        "Avg_Trade":
            result[
                "Avg_Trade"
            ],

        "Trades":
            result[
                "Trades"
            ],

    }])

    summary.to_csv(

        OUTPUT_DIR
        / "v29_results.csv",

        index=False

    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "=" * 70
    )

    print(
        "\nV29 COMPLETE"
    )


if __name__ == "__main__":

    main()

