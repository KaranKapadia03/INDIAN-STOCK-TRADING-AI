
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

OUTPUT_DIR = Path(
    "data/models/v33_stock_universe"
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

INITIAL_CAPITAL = 100000.0

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

MAX_EXPOSURE = 0.80


# ============================================================
# MODEL
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_split": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# FEATURES
# ============================================================

ALL_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Volatility_20D",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Position",
    "ATR_Percent",
    "Volume_Ratio",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


STABLE_MARKET_FEATURES = [
    "Volatility_20D",
    "Volume_Ratio",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
]


# ============================================================
# STOCK UNIVERSES
# ============================================================

ALL_20 = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
]


TOP_10 = [
    "INFY.NS",
    "TCS.NS",
    "MARUTI.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "SUNPHARMA.NS",
    "ADANIPORTS.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
]


TOP_5 = [
    "INFY.NS",
    "TCS.NS",
    "MARUTI.NS",
    "LT.NS",
    "KOTAKBANK.NS",
]


POSITIVE_ONLY = [
    "INFY.NS",
    "TCS.NS",
    "MARUTI.NS",
    "LT.NS",
    "SUNPHARMA.NS",
    "ADANIPORTS.NS",
    "KOTAKBANK.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
]


UNIVERSES = {
    "ALL_20": ALL_20,
    "TOP_10": TOP_10,
    "TOP_5": TOP_5,
    "POSITIVE_ONLY": POSITIVE_ONLY,
}


# ============================================================
# DATE
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
# LOAD DATA
# ============================================================

def load_data():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    print(
        f"Feature files found: {len(files)}"
    )

    datasets = []

    for file in files:

        symbol = file.stem.replace(
            "_features",
            ""
        )

        if symbol.endswith("_NS"):

            symbol = (
                symbol[:-3]
                + ".NS"
            )

        if symbol not in ALL_20:

            continue

        df = pd.read_csv(
            file
        )

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

        # Future 20-day stock return
        df["Future_Return_20D"] = (
            df["Close"].shift(-HORIZON)
            / df["Close"]
            - 1
        ) * 100

        # Excess return vs NIFTY
        df["Target_Excess_Return"] = (
            df["Future_Return_20D"]
            - df["NIFTY_Return_20D"]
        )

        datasets.append(
            df
        )

    if not datasets:

        raise RuntimeError(
            "No feature files loaded."
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

    return (
        data
        .sort_values(
            [
                "Date",
                "Symbol",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def generate_predictions(
    data,
    stocks,
    features
):

    universe = data[
        data["Symbol"].isin(stocks)
    ].copy()

    dates = sorted(
        universe["Date"]
        .dropna()
        .unique()
    )

    if len(dates) <= MIN_TRAIN_DAYS:

        return pd.DataFrame()

    rebalance_dates = dates[
        MIN_TRAIN_DAYS:
    ][
        ::RETRAIN_DAYS
    ]

    predictions = []

    print(
        f"Stocks: {len(stocks)}"
    )

    print(
        f"Rebalances: "
        f"{len(rebalance_dates)}"
    )

    rebalance_id = 0

    for test_date in rebalance_dates:

        test_date = pd.Timestamp(
            test_date
        )

        train_end = (
            test_date
            - pd.Timedelta(
                days=HORIZON
            )
        )

        train = universe[
            universe["Date"]
            <= train_end
        ].copy()

        test = universe[
            universe["Date"]
            == test_date
        ].copy()

        available = [
            f
            for f in features
            if f in train.columns
        ]

        if len(available) < 2:

            continue

        train = train.dropna(
            subset=[
                *available,
                "Target_Excess_Return",
            ]
        )

        test = test.dropna(
            subset=available
        )

        if train.empty or test.empty:

            continue

        model = RandomForestRegressor(
            **MODEL_PARAMS
        )

        model.fit(
            train[available],
            train[
                "Target_Excess_Return"
            ]
        )

        test = test.copy()

        test["Prediction"] = (
            model.predict(
                test[available]
            )
        )

        test["Rebalance_ID"] = (
            rebalance_id
        )

        test["Train_End"] = (
            train_end
        )

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
                    "Future_Return_20D",
                    "Target_Excess_Return",
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
            [
                "Date",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# RANK IC
# ============================================================

def rank_ic(predictions):

    values = []

    for _, group in (
        predictions.groupby("Date")
    ):

        group = group.dropna(
            subset=[
                "Prediction",
                "Target_Excess_Return",
            ]
        )

        if len(group) < 3:

            continue

        if group[
            "Prediction"
        ].nunique() < 2:

            continue

        if group[
            "Target_Excess_Return"
        ].nunique() < 2:

            continue

        ic = group[
            "Prediction"
        ].corr(
            group[
                "Target_Excess_Return"
            ],
            method="spearman"
        )

        if pd.notna(ic):

            values.append(ic)

    if not values:

        return np.nan, np.nan

    return (
        np.mean(values),
        np.median(values)
    )


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    predictions,
    data
):

    if predictions.empty:

        return None

    cash = INITIAL_CAPITAL

    positions = []

    trades = []

    equity_curve = []

    prediction_dates = sorted(
        pd.to_datetime(
            predictions["Date"]
        ).unique()
    )

    all_dates = sorted(
        pd.to_datetime(
            data["Date"]
        ).unique()
    )

    simulation_dates = [
        d
        for d in all_dates
        if d >= prediction_dates[0]
    ]

    rebalance_set = set(
        prediction_dates
    )

    for current_date in simulation_dates:

        current_date = pd.Timestamp(
            current_date
        )

        today = data[
            data["Date"]
            == current_date
        ]

        # ----------------------------------------------------
        # Update current prices
        # ----------------------------------------------------

        for position in positions:

            row = today[
                today["Symbol"]
                == position["Symbol"]
            ]

            if row.empty:

                continue

            row = row.iloc[0]

            position["Close"] = (
                row["Close"]
            )

            position["High"] = (
                row["High"]
            )

            position["Low"] = (
                row["Low"]
            )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        remaining = []

        for position in positions:

            entry = position[
                "Entry_Price"
            ]

            stop = (
                entry
                * 0.94
            )

            target = (
                entry
                * 1.12
            )

            high = position.get(
                "High",
                np.nan
            )

            low = position.get(
                "Low",
                np.nan
            )

            close = position.get(
                "Close",
                np.nan
            )

            exit_price = None
            reason = None

            # STOP FIRST
            if (
                pd.notna(low)
                and low <= stop
            ):

                exit_price = (
                    stop
                    * (1 - SLIPPAGE)
                )

                reason = "STOP"

            elif (
                pd.notna(high)
                and high >= target
            ):

                exit_price = (
                    target
                    * (1 - SLIPPAGE)
                )

                reason = "TARGET"

            elif (
                position["Days_Held"]
                >= HOLD_DAYS
                and pd.notna(close)
            ):

                exit_price = (
                    close
                    * (1 - SLIPPAGE)
                )

                reason = "TIME"

            if exit_price is None:

                position[
                    "Days_Held"
                ] += 1

                remaining.append(
                    position
                )

                continue

            gross = (
                position["Quantity"]
                * exit_price
            )

            exit_fee = (
                gross
                * TRANSACTION_COST
            )

            cash += (
                gross
                - exit_fee
            )

            invested = (
                position["Quantity"]
                * entry
            )

            total_cost = (
                invested
                + position["Entry_Fee"]
            )

            pnl = (
                gross
                - exit_fee
                - total_cost
            )

            return_pct = (
                pnl
                / total_cost
            ) * 100

            trades.append(
                {
                    "Symbol":
                        position["Symbol"],

                    "Entry_Date":
                        position["Entry_Date"],

                    "Exit_Date":
                        current_date,

                    "Entry_Price":
                        entry,

                    "Exit_Price":
                        exit_price,

                    "Quantity":
                        position["Quantity"],

                    "Return_Pct":
                        return_pct,

                    "PnL":
                        pnl,

                    "Exit_Reason":
                        reason,
                }
            )

        positions = remaining

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if current_date in rebalance_set:

            signals = predictions[
                predictions["Date"]
                == current_date
            ].copy()

            signals = (
                signals
                .sort_values(
                    "Prediction",
                    ascending=False
                )
                .head(TOP_N)
            )

            existing = {
                p["Symbol"]
                for p in positions
            }

            signals = signals[
                ~signals[
                    "Symbol"
                ].isin(existing)
            ]

            if not signals.empty:

                future_dates = [
                    d
                    for d in all_dates
                    if d > current_date
                ]

                if future_dates:

                    entry_date = (
                        pd.Timestamp(
                            future_dates[0]
                        )
                    )

                    entry_data = data[
                        data["Date"]
                        == entry_date
                    ]

                    slots = (
                        TOP_N
                        - len(existing)
                    )

                    selected = signals.head(
                        max(0, slots)
                    )

                    if not selected.empty:

                        allocation = (
                            cash
                            * MAX_EXPOSURE
                            / len(selected)
                        )

                        for _, signal in (
                            selected.iterrows()
                        ):

                            row = entry_data[
                                entry_data["Symbol"]
                                == signal["Symbol"]
                            ]

                            if row.empty:

                                continue

                            row = row.iloc[0]

                            entry_price = (
                                row["Open"]
                                * (1 + SLIPPAGE)
                            )

                            if entry_price <= 0:

                                continue

                            quantity = (
                                allocation
                                / entry_price
                            )

                            invested = (
                                quantity
                                * entry_price
                            )

                            entry_fee = (
                                invested
                                * TRANSACTION_COST
                            )

                            total = (
                                invested
                                + entry_fee
                            )

                            if total > cash:

                                continue

                            cash -= total

                            positions.append(
                                {
                                    "Symbol":
                                        signal[
                                            "Symbol"
                                        ],

                                    "Entry_Date":
                                        entry_date,

                                    "Entry_Price":
                                        entry_price,

                                    "Quantity":
                                        quantity,

                                    "Entry_Fee":
                                        entry_fee,

                                    "Days_Held":
                                        0,

                                    "Close":
                                        row["Close"],

                                    "High":
                                        row["High"],

                                    "Low":
                                        row["Low"],
                                }
                            )

        # ----------------------------------------------------
        # DAILY EQUITY
        # ----------------------------------------------------

        equity = cash

        for position in positions:

            if pd.notna(
                position.get(
                    "Close",
                    np.nan
                )
            ):

                equity += (
                    position["Quantity"]
                    * position["Close"]
                )

        equity_curve.append(
            {
                "Date":
                    current_date,

                "Equity":
                    equity,
            }
        )

    # --------------------------------------------------------
    # CLOSE REMAINING
    # --------------------------------------------------------

    if positions:

        final_date = (
            pd.Timestamp(
                simulation_dates[-1]
            )
        )

        final_data = data[
            data["Date"]
            == final_date
        ]

        for position in positions:

            row = final_data[
                final_data["Symbol"]
                == position["Symbol"]
            ]

            if row.empty:

                continue

            row = row.iloc[0]

            exit_price = (
                row["Close"]
                * (1 - SLIPPAGE)
            )

            gross = (
                position["Quantity"]
                * exit_price
            )

            exit_fee = (
                gross
                * TRANSACTION_COST
            )

            cash += (
                gross
                - exit_fee
            )

            invested = (
                position["Quantity"]
                * position["Entry_Price"]
            )

            total_cost = (
                invested
                + position["Entry_Fee"]
            )

            pnl = (
                gross
                - exit_fee
                - total_cost
            )

            return_pct = (
                pnl
                / total_cost
            ) * 100

            trades.append(
                {
                    "Symbol":
                        position["Symbol"],

                    "Entry_Date":
                        position["Entry_Date"],

                    "Exit_Date":
                        final_date,

                    "Entry_Price":
                        position["Entry_Price"],

                    "Exit_Price":
                        exit_price,

                    "Quantity":
                        position["Quantity"],

                    "Return_Pct":
                        return_pct,

                    "PnL":
                        pnl,

                    "Exit_Reason":
                        "END",
                }
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    if equity_df.empty:

        return None

    final_equity = (
        equity_df["Equity"].iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    ) * 100

    elapsed_days = max(
        1,
        (
            equity_df["Date"].iloc[-1]
            - equity_df["Date"].iloc[0]
        ).days
    )

    cagr = (
        (
            final_equity
            / INITIAL_CAPITAL
        )
        ** (
            365
            / elapsed_days
        )
        - 1
    ) * 100

    running_max = (
        equity_df["Equity"]
        .cummax()
    )

    drawdown = (
        equity_df["Equity"]
        / running_max
        - 1
    )

    max_dd = (
        drawdown.min()
        * 100
    )

    daily_returns = (
        equity_df["Equity"]
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

    if trades_df.empty:

        win_rate = np.nan
        profit_factor = np.nan
        avg_trade = np.nan

    else:

        win_rate = (
            (
                trades_df["Return_Pct"]
                > 0
            ).mean()
            * 100
        )

        avg_trade = (
            trades_df["Return_Pct"]
            .mean()
        )

        gross_profit = (
            trades_df.loc[
                trades_df["PnL"] > 0,
                "PnL"
            ].sum()
        )

        gross_loss = abs(
            trades_df.loc[
                trades_df["PnL"] < 0,
                "PnL"
            ].sum()
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else np.inf
        )

    return {
        "Final_Equity":
            final_equity,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            max_dd,

        "Daily_Sharpe":
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
        "V33 - STOCK UNIVERSE STABILITY TEST"
    )

    print("=" * 70)

    data = load_data()

    print(
        f"Market rows: "
        f"{len(data):,}"
    )

    results = []

    for universe_name, stocks in (
        UNIVERSES.items()
    ):

        for feature_name, features in [
            (
                "ALL_FEATURES",
                ALL_FEATURES
            ),
            (
                "STABLE_MARKET",
                STABLE_MARKET_FEATURES
            ),
        ]:

            print("\n")
            print("-" * 70)

            print(
                f"TEST: "
                f"{universe_name} | "
                f"{feature_name}"
            )

            print("-" * 70)

            predictions = (
                generate_predictions(
                    data,
                    stocks,
                    features
                )
            )

            if predictions.empty:

                print(
                    "No predictions."
                )

                continue

            mean_ic, median_ic = (
                rank_ic(
                    predictions
                )
            )

            result = backtest(
                predictions,
                data
            )

            if result is None:

                continue

            print(
                f"Predictions: "
                f"{len(predictions):,}"
            )

            print(
                f"Rebalances: "
                f"{predictions['Date'].nunique()}"
            )

            print(
                f"Final Equity: "
                f"₹{result['Final_Equity']:,.2f}"
            )

            print(
                f"Return: "
                f"{result['Total_Return']:.2f}%"
            )

            print(
                f"CAGR: "
                f"{result['CAGR']:.2f}%"
            )

            print(
                f"Max DD: "
                f"{result['Max_Drawdown']:.2f}%"
            )

            print(
                f"Daily Sharpe: "
                f"{result['Daily_Sharpe']:.2f}"
            )

            print(
                f"Win Rate: "
                f"{result['Win_Rate']:.2f}%"
            )

            print(
                f"Profit Factor: "
                f"{result['Profit_Factor']:.2f}"
            )

            print(
                f"Avg Trade: "
                f"{result['Avg_Trade']:.3f}%"
            )

            print(
                f"Trades: "
                f"{result['Trades']}"
            )

            print(
                f"Mean Rank IC: "
                f"{mean_ic:.4f}"
            )

            prefix = (
                f"{universe_name}_"
                f"{feature_name}"
            )

            predictions.to_csv(
                OUTPUT_DIR
                / f"{prefix}_predictions.csv",
                index=False
            )

            result[
                "Trades_Data"
            ].to_csv(
                OUTPUT_DIR
                / f"{prefix}_trades.csv",
                index=False
            )

            result[
                "Equity"
            ].to_csv(
                OUTPUT_DIR
                / f"{prefix}_equity.csv",
                index=False
            )

            results.append(
                {
                    "Universe":
                        universe_name,

                    "Feature_Set":
                        feature_name,

                    "Stocks":
                        len(stocks),

                    "Predictions":
                        len(predictions),

                    "Rebalances":
                        predictions[
                            "Date"
                        ].nunique(),

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

                    "Daily_Sharpe":
                        result[
                            "Daily_Sharpe"
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

                    "Mean_Rank_IC":
                        mean_ic,

                    "Median_Rank_IC":
                        median_ic,
                }
            )

    if not results:

        raise RuntimeError(
            "No V33 results generated."
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "Total_Return",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n")
    print("=" * 70)

    print(
        "V33 FINAL COMPARISON"
    )

    print("=" * 70)

    print(
        results_df[
            [
                "Universe",
                "Feature_Set",
                "Stocks",
                "Total_Return",
                "CAGR",
                "Max_Drawdown",
                "Daily_Sharpe",
                "Win_Rate",
                "Profit_Factor",
                "Avg_Trade",
                "Trades",
                "Mean_Rank_IC",
            ]
        ].to_string(
            index=False
        )
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "v33_comparison.csv",
        index=False
    )

    print("\n")
    print("=" * 70)

    print(
        "FILES SAVED"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print("=" * 70)

    print(
        "\nV33 COMPLETE"
    )


if __name__ == "__main__":
    main()

