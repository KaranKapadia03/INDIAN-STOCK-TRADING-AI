
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

OUTPUT_DIR = Path(
    "data/models/v34_exit_architecture"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Use the V33 Stable Market predictions.
# V33 already contains the walk-forward predictions
# and actual future returns.
PREDICTION_FILE = Path(
    "data/models/v33_stock_universe/"
    "ALL_20_STABLE_MARKET_predictions.csv"
)

HORIZON = 20
TOP_N = 3

INITIAL_CAPITAL = 100000.0

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

MAX_EXPOSURE = 0.80


# ============================================================
# EXIT CONFIGURATIONS
# ============================================================

EXIT_CONFIGS = {

    "BASE_6_12": {
        "stop": 0.06,
        "target": 0.12,
        "max_hold": 20,
    },

    "WIDE_8_12": {
        "stop": 0.08,
        "target": 0.12,
        "max_hold": 20,
    },

    "WIDE_10_15": {
        "stop": 0.10,
        "target": 0.15,
        "max_hold": 20,
    },

    "NO_STOP_12_TARGET": {
        "stop": None,
        "target": 0.12,
        "max_hold": 20,
    },

    "PURE_20D_HOLD": {
        "stop": None,
        "target": None,
        "max_hold": 20,
    },

}


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
# LOAD FEATURE DATA
# ============================================================

def load_market_data():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    datasets = []

    for file in files:

        symbol = (
            file.stem
            .replace(
                "_features",
                ""
            )
        )

        if symbol.endswith("_NS"):

            symbol = (
                symbol[:-3]
                + ".NS"
            )

        df = pd.read_csv(
            file
        )

        df = normalize_dates(
            df
        )

        df["Symbol"] = symbol

        datasets.append(
            df[
                [
                    "Date",
                    "Symbol",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            ]
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    return (
        data
        .drop_duplicates(
            [
                "Date",
                "Symbol",
            ]
        )
        .sort_values(
            [
                "Date",
                "Symbol",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions = normalize_dates(
        predictions
    )

    required = [
        "Date",
        "Symbol",
        "Prediction",
    ]

    missing = [
        x
        for x in required
        if x not in predictions.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    return (
        predictions
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
# BACKTEST
# ============================================================

def backtest(
    predictions,
    market_data,
    config,
):

    stop_pct = config["stop"]
    target_pct = config["target"]
    max_hold = config["max_hold"]

    prediction_dates = sorted(
        pd.to_datetime(
            predictions["Date"]
        ).unique()
    )

    all_dates = sorted(
        pd.to_datetime(
            market_data["Date"]
        ).unique()
    )

    simulation_dates = [
        x
        for x in all_dates
        if x >= prediction_dates[0]
    ]

    rebalance_dates = set(
        prediction_dates
    )

    cash = INITIAL_CAPITAL

    positions = []

    trades = []

    equity_curve = []

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for current_date in simulation_dates:

        current_date = pd.Timestamp(
            current_date
        )

        today = market_data[
            market_data["Date"]
            == current_date
        ]

        # ----------------------------------------------------
        # UPDATE POSITIONS
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
        # EXIT POSITIONS
        # ----------------------------------------------------

        remaining = []

        for position in positions:

            entry_price = (
                position["Entry_Price"]
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

            stop_price = None

            target_price = None

            if stop_pct is not None:

                stop_price = (
                    entry_price
                    * (1 - stop_pct)
                )

            if target_pct is not None:

                target_price = (
                    entry_price
                    * (1 + target_pct)
                )

            exit_price = None
            reason = None

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            if (
                stop_price is not None
                and pd.notna(low)
                and low <= stop_price
            ):

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE)
                )

                reason = "STOP"

            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            elif (
                target_price is not None
                and pd.notna(high)
                and high >= target_price
            ):

                exit_price = (
                    target_price
                    * (1 - SLIPPAGE)
                )

                reason = "TARGET"

            # ------------------------------------------------
            # MAX HOLD
            # ------------------------------------------------

            elif (
                position["Days_Held"]
                >= max_hold
                and pd.notna(close)
            ):

                exit_price = (
                    close
                    * (1 - SLIPPAGE)
                )

                reason = "TIME"

            # ------------------------------------------------
            # KEEP POSITION
            # ------------------------------------------------

            if exit_price is None:

                position[
                    "Days_Held"
                ] += 1

                remaining.append(
                    position
                )

                continue

            # ------------------------------------------------
            # CALCULATE PNL
            # ------------------------------------------------

            gross_value = (
                position["Quantity"]
                * exit_price
            )

            exit_fee = (
                gross_value
                * TRANSACTION_COST
            )

            cash += (
                gross_value
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
                gross_value
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
                        reason,
                }
            )

        positions = remaining

        # ----------------------------------------------------
        # NEW POSITIONS
        # ----------------------------------------------------

        if current_date in rebalance_dates:

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

            existing_symbols = {
                p["Symbol"]
                for p in positions
            }

            signals = signals[
                ~signals[
                    "Symbol"
                ].isin(
                    existing_symbols
                )
            ]

            slots = (
                TOP_N
                - len(existing_symbols)
            )

            signals = signals.head(
                max(0, slots)
            )

            if not signals.empty:

                future_dates = [
                    x
                    for x in all_dates
                    if x > current_date
                ]

                if future_dates:

                    entry_date = (
                        pd.Timestamp(
                            future_dates[0]
                        )
                    )

                    entry_data = market_data[
                        market_data["Date"]
                        == entry_date
                    ]

                    allocation = (
                        cash
                        * MAX_EXPOSURE
                        / len(signals)
                    )

                    for _, signal in (
                        signals.iterrows()
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

                        total_cost = (
                            invested
                            + entry_fee
                        )

                        if total_cost > cash:

                            continue

                        cash -= total_cost

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

            close = position.get(
                "Close",
                np.nan
            )

            if pd.notna(close):

                equity += (
                    position["Quantity"]
                    * close
                )

        equity_curve.append(
            {
                "Date":
                    current_date,

                "Equity":
                    equity,
            }
        )

    # ========================================================
    # FORCE CLOSE
    # ========================================================

    if positions:

        final_date = (
            pd.Timestamp(
                simulation_dates[-1]
            )
        )

        final_data = market_data[
            market_data["Date"]
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

            gross_value = (
                position["Quantity"]
                * exit_price
            )

            exit_fee = (
                gross_value
                * TRANSACTION_COST
            )

            cash += (
                gross_value
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
                gross_value
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

    # ========================================================
    # METRICS
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

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
            365 / elapsed_days
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
        avg_trade = np.nan
        profit_factor = np.nan

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

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

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
        "V34 - EXIT ARCHITECTURE TEST"
    )

    print("=" * 70)

    predictions = load_predictions()

    market_data = load_market_data()

    print(
        f"Predictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Market rows: "
        f"{len(market_data):,}"
    )

    results = []

    for name, config in (
        EXIT_CONFIGS.items()
    ):

        print("\n")
        print("-" * 70)

        print(
            f"TEST: {name}"
        )

        print("-" * 70)

        result = backtest(
            predictions,
            market_data,
            config
        )

        if result is None:
            continue

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

        trades = result[
            "Trades_Data"
        ]

        if not trades.empty:

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

                    Avg_Return=(
                        "Return_Pct",
                        "mean"
                    ),

                    Total_PnL=(
                        "PnL",
                        "sum"
                    ),

                    Win_Rate=(
                        "Return_Pct",
                        lambda x:
                        (
                            x > 0
                        ).mean()
                        * 100
                    ),
                )
                .reset_index()
            )

            print(
                "\nExit analysis:"
            )

            print(
                exit_summary.to_string(
                    index=False
                )
            )

        prefix = (
            f"{name}"
        )

        trades.to_csv(
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
                "Exit_Strategy":
                    name,

                "Stop":
                    config["stop"],

                "Target":
                    config["target"],

                "Max_Hold":
                    config["max_hold"],

                "Final_Equity":
                    result[
                        "Final_Equity"
                    ],

                "Total_Return":
                    result[
                        "Total_Return"
                    ],

                "CAGR":
                    result["CAGR"],

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
            }
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison = (
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
        "V34 FINAL COMPARISON"
    )

    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "v34_comparison.csv",
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
        "\nV34 COMPLETE"
    )


if __name__ == "__main__":
    main()

