
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

OUTPUT_DIR = Path(
    "data/models/v28_out_of_sample"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

MIN_TRAIN_DAYS = 500

RETRAIN_DAYS = 20

TOP_N = 3

HOLD_DAYS = 20

INITIAL_CAPITAL = 100_000

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

RANDOM_STATE = 42


# ============================================================
# LOCKED FEATURE SETS
# ============================================================

# This is the V27 winner.
# DO NOT MODIFY DURING V28.

STABLE_MARKET_FEATURES = [

    "Volatility_20D",

    "Volume_Ratio",

    "NIFTY_Return_5D",

    "NIFTY_Return_20D",

    "BANKNIFTY_Return_5D",

    "BANKNIFTY_Return_20D",
]


# V24 benchmark

V24_FEATURES = [

    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    "SMA_20",
    "SMA_50",
    "EMA_20",

    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",

    "RSI_14",

    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    "BB_Position",

    "Volatility_20D",
    "ATR_Percent",

    "Volume_Ratio",

    "NIFTY_Return_5D",
    "NIFTY_Return_20D",

    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",

    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
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
# SYMBOL
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
# LOAD DATA
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

            "Close",

            "NIFTY_Return_20D",

        ]

        missing = [

            column

            for column in required

            if column not in df.columns

        ]

        if missing:

            print(
                f"Skipping {symbol}: "
                f"{missing}"
            )

            continue

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Future stock return
        # ----------------------------------------------------

        df[
            "Stock_Return_20D"
        ] = (

            df["Close"].shift(-HORIZON)
            / df["Close"]
            - 1

        ) * 100

        # ----------------------------------------------------
        # Excess return vs NIFTY
        # ----------------------------------------------------

        df[
            "Excess_Return_20D"
        ] = (

            df["Stock_Return_20D"]

            - df["NIFTY_Return_20D"]

        )

        df[
            "Symbol"
        ] = symbol

        datasets.append(
            df
        )

    if not datasets:

        raise RuntimeError(
            "No usable datasets."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    data = data.loc[
        :,
        ~data.columns.duplicated()
    ]

    data = (
        data
        .sort_values(
            [
                "Date",
                "Symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    data,
    features
):

    features = list(
        dict.fromkeys(
            features
        )
    )

    available = [

        feature

        for feature in features

        if feature in data.columns

    ]

    missing = [

        feature

        for feature in features

        if feature not in data.columns

    ]

    if missing:

        print(
            "Missing features:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

    if not available:

        return None, []

    # --------------------------------------------------------
    # Base columns deliberately exclude
    # NIFTY_Return_20D.
    # --------------------------------------------------------

    base_columns = [

        "Date",

        "Symbol",

        "Close",

        "Stock_Return_20D",

        "Excess_Return_20D",

    ]

    columns = list(
        dict.fromkeys(
            base_columns
            + available
        )
    )

    df = data[
        columns
    ].copy()

    # --------------------------------------------------------
    # Cross-sectional fill
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

    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    for feature in available:

        median_value = (
            df[feature].median()
        )

        if pd.notna(
            median_value
        ):

            df[feature] = (
                df[feature]
                .fillna(
                    median_value
                )
            )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Excess_Return_20D"
        ]
        + available
    )

    return (
        df
        .sort_values(
            [
                "Date",
                "Symbol",
            ]
        )
        .reset_index(
            drop=True
        ),
        available
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def train_model(
    train,
    features
):

    model = RandomForestRegressor(

        n_estimators=400,

        max_depth=8,

        min_samples_split=15,

        min_samples_leaf=5,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )

    model.fit(

        train[features],

        train[
            "Excess_Return_20D"
        ],

    )

    return model


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward(
    data,
    features
):

    dates = sorted(
        data[
            "Date"
        ].unique()
    )

    predictions = []

    rebalance_number = 0

    for i in range(

        MIN_TRAIN_DAYS,

        len(dates)
        - HORIZON,

        RETRAIN_DAYS,

    ):

        test_date = dates[i]

        # ----------------------------------------------------
        # Strict leakage protection
        # ----------------------------------------------------

        cutoff_index = (
            i - HORIZON
        )

        cutoff_date = dates[
            cutoff_index
        ]

        train = data[
            data["Date"]
            <= cutoff_date
        ].copy()

        test = data[
            data["Date"]
            == test_date
        ].copy()

        if train.empty:

            continue

        if test.empty:

            continue

        if (
            train["Date"].nunique()
            < MIN_TRAIN_DAYS
        ):

            continue

        model = train_model(
            train,
            features
        )

        test = test.copy()

        test[
            "Prediction"
        ] = model.predict(
            test[
                features
            ]
        )

        test[
            "Rebalance"
        ] = (
            rebalance_number
        )

        test[
            "Train_End"
        ] = cutoff_date

        predictions.append(

            test[
                [
                    "Date",

                    "Symbol",

                    "Close",

                    "Stock_Return_20D",

                    "NIFTY_Return_20D",

                    "Excess_Return_20D",

                    "Prediction",

                    "Rebalance",

                    "Train_End",

                ]
            ].copy()

        )

        rebalance_number += 1

    if not predictions:

        return pd.DataFrame()

    return (

        pd.concat(
            predictions,
            ignore_index=True
        )

        .sort_values(
            [
                "Date",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ]
        )

        .reset_index(
            drop=True
        )

    )


# ============================================================
# RANK IC
# ============================================================

def calculate_rank_ic(
    predictions
):

    values = []

    for date, group in (
        predictions.groupby(
            "Date"
        )
    ):

        if len(group) < 3:

            continue

        if (
            group[
                "Prediction"
            ].nunique()
            < 2
        ):

            continue

        if (
            group[
                "Excess_Return_20D"
            ].nunique()
            < 2
        ):

            continue

        ic = group[
            "Prediction"
        ].corr(

            group[
                "Excess_Return_20D"
            ],

            method="spearman"

        )

        if pd.notna(ic):

            values.append(
                ic
            )

    if not values:

        return (
            np.nan,
            np.nan
        )

    return (

        float(
            np.mean(values)
        ),

        float(
            np.median(values)
        ),

    )


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def portfolio_backtest(
    predictions
):

    if predictions.empty:

        return None

    dates = sorted(
        predictions[
            "Date"
        ].unique()
    )

    capital = (
        INITIAL_CAPITAL
    )

    equity_rows = []

    trade_rows = []

    for date in dates:

        day = predictions[
            predictions["Date"]
            == date
        ].copy()

        if day.empty:

            continue

        selected = (

            day

            .sort_values(
                "Prediction",
                ascending=False
            )

            .head(TOP_N)

        )

        if selected.empty:

            continue

        position_weight = (
            0.80
            / len(selected)
        )

        portfolio_return = 0.0

        capital_before = capital

        for _, row in (
            selected.iterrows()
        ):

            gross_return = (
                row[
                    "Stock_Return_20D"
                ]
                / 100
            )

            round_trip_cost = (

                2
                * (
                    TRANSACTION_COST
                    + SLIPPAGE
                )

            )

            net_return = (

                (
                    1
                    + gross_return
                )

                * (

                    1
                    - round_trip_cost

                )

                - 1

            )

            contribution = (

                position_weight
                * net_return

            )

            portfolio_return += (
                contribution
            )

            trade_rows.append({

                "Date":
                    date,

                "Symbol":
                    row[
                        "Symbol"
                    ],

                "Prediction":
                    row[
                        "Prediction"
                    ],

                "Stock_Return":
                    gross_return
                    * 100,

                "Net_Return":
                    net_return
                    * 100,

                "Position_Weight":
                    position_weight,

                "PnL":
                    capital_before
                    * contribution,

            })

        capital *= (
            1
            + portfolio_return
        )

        equity_rows.append({

            "Date":
                date,

            "Equity":
                capital,

            "Portfolio_Return":
                portfolio_return
                * 100,

        })

    if not equity_rows:

        return None

    equity = pd.DataFrame(
        equity_rows
    )

    trades = pd.DataFrame(
        trade_rows
    )

    equity[
        "Equity"
    ] = pd.to_numeric(
        equity[
            "Equity"
        ],
        errors="coerce"
    )

    equity = equity.dropna(
        subset=[
            "Equity"
        ]
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    equity[
        "Peak"
    ] = (
        equity[
            "Equity"
        ]
        .cummax()
    )

    equity[
        "Drawdown"
    ] = (

        equity[
            "Equity"
        ]

        /

        equity[
            "Peak"
        ]

        - 1

    )

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    total_return = (

        capital
        / INITIAL_CAPITAL
        - 1

    )

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    start_date = pd.Timestamp(
        equity[
            "Date"
        ].iloc[0]
    )

    end_date = pd.Timestamp(
        equity[
            "Date"
        ].iloc[-1]
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
            capital
            / INITIAL_CAPITAL
        )

        ** (
            1 / years
        )

        - 1

    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    returns = (

        equity[
            "Portfolio_Return"
        ]

        / 100

    )

    if returns.std() > 0:

        sharpe = (

            returns.mean()
            / returns.std()
            * np.sqrt(
                252 / HOLD_DAYS
            )

        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Trade statistics
    # --------------------------------------------------------

    trade_count = len(
        trades
    )

    if trade_count:

        win_rate = (

            (
                trades[
                    "Net_Return"
                ]
                > 0
            ).mean()

        )

        avg_trade = (

            trades[
                "Net_Return"
            ].mean()
            / 100

        )

        gross_profit = (

            trades.loc[
                trades[
                    "Net_Return"
                ] > 0,
                "Net_Return"
            ].sum()

        )

        gross_loss = abs(

            trades.loc[
                trades[
                    "Net_Return"
                ] < 0,
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

    else:

        win_rate = np.nan

        avg_trade = np.nan

        profit_factor = np.nan

    return {

        "Final_Equity":
            capital,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            equity[
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
            trade_count,

        "Rebalances":
            len(equity),

        "Equity":
            equity,

        "Trades_Data":
            trades,

    }


# ============================================================
# NIFTY BUY-AND-HOLD
# ============================================================

def nifty_benchmark(
    data,
    start_date,
    end_date
):

    benchmark = data[
        (
            data["Date"]
            >= start_date
        )
        &
        (
            data["Date"]
            <= end_date
        )
    ].copy()

    # NIFTY return is available as a
    # daily market-context return.
    #
    # We reconstruct the NIFTY level from
    # the cumulative daily return.

    daily = (

        benchmark[
            [
                "Date",
                "NIFTY_Return_20D"
            ]
        ]

        .drop_duplicates(
            "Date"
        )

        .sort_values(
            "Date"
        )

    )

    if daily.empty:

        return np.nan

    # This is only a reference approximation.
    # It is NOT used by the model.

    return np.nan


# ============================================================
# RUN MODEL
# ============================================================

def run_model(
    name,
    data,
    features
):

    print("\n")
    print("=" * 70)
    print(
        f"RUNNING {name}"
    )
    print("=" * 70)

    prepared, actual_features = (
        prepare_data(
            data,
            features
        )
    )

    if prepared is None:

        print(
            "No usable data."
        )

        return None

    print(
        f"Features: "
        f"{len(actual_features)}"
    )

    print(
        ", ".join(
            actual_features
        )
    )

    predictions = walk_forward(
        prepared,
        actual_features
    )

    if predictions.empty:

        print(
            "No predictions."
        )

        return None

    result = portfolio_backtest(
        predictions
    )

    if result is None:

        return None

    mean_ic, median_ic = (
        calculate_rank_ic(
            predictions
        )
    )

    print(
        f"\nPredictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Rebalances: "
        f"{result['Rebalances']}"
    )

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
        f"Mean Rank IC: "
        f"{mean_ic:.4f}"
    )

    print(
        f"Median Rank IC: "
        f"{median_ic:.4f}"
    )

    return {

        "Model":
            name,

        "Feature_Count":
            len(actual_features),

        "Predictions":
            len(predictions),

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

        "Rebalances":
            result[
                "Rebalances"
            ],

        "Mean_Rank_IC":
            mean_ic,

        "Median_Rank_IC":
            median_ic,

        "Predictions_Data":
            predictions,

        "Equity_Data":
            result[
                "Equity"
            ],

        "Trades_Data":
            result[
                "Trades_Data"
            ],

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
        "V28 - OUT-OF-SAMPLE VALIDATION"
    )
    print("=" * 70)

    data = load_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # LOCKED EXPERIMENTS
    # --------------------------------------------------------

    experiments = {

        "STABLE_MARKET":
            STABLE_MARKET_FEATURES,

        "V24_ALL":
            V24_FEATURES,

    }

    results = []

    all_predictions = []

    for name, features in (
        experiments.items()
    ):

        result = run_model(
            name,
            data,
            features
        )

        if result is None:

            continue

        results.append({

            key: value

            for key, value
            in result.items()

            if key not in [

                "Predictions_Data",
                "Equity_Data",
                "Trades_Data",

            ]

        })

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        prediction_data = (
            result[
                "Predictions_Data"
            ].copy()
        )

        prediction_data[
            "Model"
        ] = name

        all_predictions.append(
            prediction_data
        )

        safe_name = (
            name.lower()
        )

        result[
            "Predictions_Data"
        ].to_csv(

            OUTPUT_DIR
            / f"{safe_name}_predictions.csv",

            index=False

        )

        result[
            "Equity_Data"
        ].to_csv(

            OUTPUT_DIR
            / f"{safe_name}_equity.csv",

            index=False

        )

        result[
            "Trades_Data"
        ].to_csv(

            OUTPUT_DIR
            / f"{safe_name}_trades.csv",

            index=False

        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No validation models completed."
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "Total_Return",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 70)
    print(
        "V28 OUT-OF-SAMPLE COMPARISON"
    )
    print("=" * 70)

    display_df = (
        results_df.copy()
    )

    for column in [

        "Total_Return",
        "CAGR",
        "Max_Drawdown",
        "Win_Rate",
        "Avg_Trade",

    ]:

        display_df[
            column
        ] = (
            display_df[
                column
            ]
            * 100
        )

    print(
        display_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results_file = (
        OUTPUT_DIR
        / "v28_validation_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    if all_predictions:

        combined_predictions = (
            pd.concat(
                all_predictions,
                ignore_index=True
            )
        )

        combined_file = (
            OUTPUT_DIR
            / "v28_all_predictions.csv"
        )

        combined_predictions.to_csv(
            combined_file,
            index=False
        )

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    best = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print(
        "V28 RESULT"
    )
    print("=" * 70)

    print(
        f"Best Model: "
        f"{best['Model']}"
    )

    print(
        f"Return: "
        f"{best['Total_Return'] * 100:.2f}%"
    )

    print(
        f"CAGR: "
        f"{best['CAGR'] * 100:.2f}%"
    )

    print(
        f"Sharpe: "
        f"{best['Sharpe']:.2f}"
    )

    print(
        f"Max DD: "
        f"{best['Max_Drawdown'] * 100:.2f}%"
    )

    print("\n")
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(
        results_file.resolve()
    )

    if all_predictions:

        print(
            combined_file.resolve()
        )

    print("\n")
    print("=" * 70)
    print(
        "V28 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()

