
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/processed")

PREDICTION_FILE = Path(
    "data/models/v29_realistic_backtest/"
    "v29_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/models/v30_daily_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_CAPITAL = 100_000.0

TOP_N = 3

MAX_EXPOSURE = 0.80

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

STOP_LOSS = 0.06

TAKE_PROFIT = 0.12

MAX_HOLD_DAYS = 20


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
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    files = sorted(
        DATA_DIR.glob(
            "*_features.csv"
        )
    )

    print(
        f"Market datasets: "
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

        df["Symbol"] = symbol

        df = (
            df
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

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

    if not datasets:

        raise RuntimeError(
            "No market data found."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
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

    predictions = (
        predictions
        .sort_values(
            [
                "Date",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return predictions


# ============================================================
# CREATE DAILY PRICE MAP
# ============================================================

def create_price_map(
    market_data
):

    price_map = {}

    for symbol, group in (
        market_data.groupby(
            "Symbol"
        )
    ):

        group = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

        price_map[symbol] = group

    return price_map


# ============================================================
# GET NEXT TRADING DAY
# ============================================================

def get_next_day(
    dates,
    current_date
):

    index = dates.searchsorted(
        current_date
    )

    next_index = index + 1

    if next_index >= len(dates):

        return None

    return dates[next_index]


# ============================================================
# DAILY BACKTEST ENGINE
# ============================================================

def run_backtest(
    predictions,
    market_data
):

    all_dates = pd.DatetimeIndex(
        sorted(
            market_data[
                "Date"
            ].unique()
        )
    )

    price_map = create_price_map(
        market_data
    )

    prediction_dates = set(
        predictions[
            "Date"
        ].unique()
    )

    cash = (
        INITIAL_CAPITAL
    )

    positions = {}

    trades = []

    equity_records = []

    portfolio_records = []

    last_rebalance_date = None

    # --------------------------------------------------------
    # Process EVERY trading day
    # --------------------------------------------------------

    for current_date in all_dates:

        # ====================================================
        # 1. PROCESS EXISTING POSITIONS
        # ====================================================

        symbols_to_remove = []

        for symbol, position in (
            positions.items()
        ):

            stock_data = price_map.get(
                symbol
            )

            if stock_data is None:
                continue

            today = stock_data[
                stock_data["Date"]
                == current_date
            ]

            if today.empty:
                continue

            row = today.iloc[0]

            entry_date = position[
                "Entry_Date"
            ]

            entry_price = position[
                "Entry_Price"
            ]

            stop_price = (
                entry_price
                * (
                    1 - STOP_LOSS
                )
            )

            target_price = (
                entry_price
                * (
                    1 + TAKE_PROFIT
                )
            )

            held_days = (
                all_dates.get_loc(
                    current_date
                )
                -
                all_dates.get_loc(
                    entry_date
                )
            )

            exit_reason = None

            exit_price = None

            # ------------------------------------------------
            # Intraday stop
            # ------------------------------------------------

            if row["Low"] <= stop_price:

                exit_reason = "STOP"

                exit_price = (
                    stop_price
                    * (
                        1 - SLIPPAGE
                    )
                )

            # ------------------------------------------------
            # Intraday target
            # ------------------------------------------------

            elif (
                row["High"]
                >= target_price
            ):

                exit_reason = "TARGET"

                exit_price = (
                    target_price
                    * (
                        1 - SLIPPAGE
                    )
                )

            # ------------------------------------------------
            # Maximum holding period
            # ------------------------------------------------

            elif (
                held_days
                >= MAX_HOLD_DAYS
            ):

                exit_reason = "TIME"

                exit_price = (
                    row["Close"]
                    * (
                        1 - SLIPPAGE
                    )
                )

            if exit_reason is None:

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

            trade_return = (
                pnl
                / entry_value
            )

            cash += proceeds

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
                    trade_return,

                "Hold_Days":
                    held_days,

            })

            symbols_to_remove.append(
                symbol
            )

        for symbol in symbols_to_remove:

            del positions[
                symbol
            ]

        # ====================================================
        # 2. REBALANCE
        # ====================================================

        if current_date in prediction_dates:

            day_predictions = (
                predictions[
                    predictions["Date"]
                    == current_date
                ]
                .sort_values(
                    "Prediction",
                    ascending=False
                )
                .copy()
            )

            selected = (
                day_predictions
                .head(TOP_N)
            )

            # ------------------------------------------------
            # Existing symbols are not duplicated.
            # ------------------------------------------------

            selected = selected[
                ~selected["Symbol"]
                .isin(
                    positions.keys()
                )
            ]

            if not selected.empty:

                available_cash = (
                    cash
                    * MAX_EXPOSURE
                )

                position_value = (
                    available_cash
                    / TOP_N
                )

                for _, prediction in (
                    selected.iterrows()
                ):

                    symbol = prediction[
                        "Symbol"
                    ]

                    next_date = (
                        get_next_day(
                            all_dates,
                            current_date
                        )
                    )

                    if next_date is None:

                        continue

                    stock_data = (
                        price_map.get(
                            symbol
                        )
                    )

                    if stock_data is None:

                        continue

                    entry_row = stock_data[
                        stock_data["Date"]
                        == next_date
                    ]

                    if entry_row.empty:

                        continue

                    entry_row = (
                        entry_row.iloc[0]
                    )

                    entry_price = (
                        entry_row["Open"]
                        * (
                            1 + SLIPPAGE
                        )
                    )

                    entry_cost = (
                        position_value
                        * TRANSACTION_COST
                    )

                    required_cash = (
                        position_value
                        + entry_cost
                    )

                    if (
                        required_cash
                        > cash
                    ):

                        continue

                    shares = (
                        position_value
                        / entry_price
                    )

                    cash -= (
                        required_cash
                    )

                    positions[
                        symbol
                    ] = {

                        "Symbol":
                            symbol,

                        "Entry_Date":
                            next_date,

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

                    }

            last_rebalance_date = (
                current_date
            )

        # ====================================================
        # 3. DAILY MARK-TO-MARKET
        # ====================================================

        equity = cash

        position_value_total = 0.0

        for symbol, position in (
            positions.items()
        ):

            stock_data = price_map.get(
                symbol
            )

            if stock_data is None:

                continue

            today = stock_data[
                stock_data["Date"]
                == current_date
            ]

            if today.empty:

                continue

            close = (
                today.iloc[0]["Close"]
            )

            market_value = (
                position["Shares"]
                * close
            )

            position_value_total += (
                market_value
            )

            equity += market_value

        equity_records.append({

            "Date":
                current_date,

            "Equity":
                equity,

            "Cash":
                cash,

            "Invested":
                position_value_total,

            "Positions":
                len(positions),

        })

    # ========================================================
    # FORCE CLOSE REMAINING POSITIONS
    # ========================================================

    if positions:

        final_date = all_dates[-1]

        for symbol, position in (
            list(positions.items())
        ):

            stock_data = price_map.get(
                symbol
            )

            if stock_data is None:
                continue

            row = stock_data[
                stock_data["Date"]
                == final_date
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            exit_price = (
                row["Close"]
                * (
                    1 - SLIPPAGE
                )
            )

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

            trade_return = (
                pnl
                / entry_value
            )

            cash += proceeds

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

                "Prediction":
                    position[
                        "Prediction"
                    ],

                "Exit_Reason":
                    "END",

                "Entry_Value":
                    entry_value,

                "Exit_Value":
                    proceeds,

                "PnL":
                    pnl,

                "Return":
                    trade_return,

                "Hold_Days":
                    (
                        all_dates.get_loc(
                            final_date
                        )
                        -
                        all_dates.get_loc(
                            position[
                                "Entry_Date"
                            ]
                        )
                    ),

            })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    equity = pd.DataFrame(
        equity_records
    )

    trades_df = pd.DataFrame(
        trades
    )

    if equity.empty:

        raise RuntimeError(
            "No equity data generated."
        )

    equity = (
        equity
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # ========================================================
    # DAILY RETURNS
    # ========================================================

    equity[
        "Daily_Return"
    ] = (
        equity[
            "Equity"
        ]
        .pct_change()
        .fillna(0)
    )

    # ========================================================
    # EQUITY PEAK / DRAWDOWN
    # ========================================================

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

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    final_equity = (
        equity[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

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
            final_equity
            / INITIAL_CAPITAL
        )
        ** (
            1 / years
        )
        - 1

    )

    # --------------------------------------------------------
    # TRUE DAILY SHARPE
    # --------------------------------------------------------

    daily_returns = (
        equity[
            "Daily_Return"
        ]
    )

    if daily_returns.std() > 0:

        sharpe = (

            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)

        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Sortino
    # --------------------------------------------------------

    downside = (
        daily_returns[
            daily_returns < 0
        ]
    )

    if (
        len(downside) > 0
        and downside.std() > 0
    ):

        sortino = (

            daily_returns.mean()
            / downside.std()
            * np.sqrt(252)

        )

    else:

        sortino = np.nan

    # --------------------------------------------------------
    # Calmar
    # --------------------------------------------------------

    max_drawdown = (
        equity[
            "Drawdown"
        ].min()
    )

    if max_drawdown < 0:

        calmar = (
            cagr
            / abs(max_drawdown)
        )

    else:

        calmar = np.nan

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    volatility = (
        daily_returns.std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # Trade metrics
    # --------------------------------------------------------

    if not trades_df.empty:

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

        best_trade = (
            trades_df[
                "Return"
            ].max()
        )

        worst_trade = (
            trades_df[
                "Return"
            ].min()
        )

    else:

        win_rate = np.nan
        avg_trade = np.nan
        profit_factor = np.nan
        best_trade = np.nan
        worst_trade = np.nan

    # ========================================================
    # MONTHLY RETURNS
    # ========================================================

    monthly = (
        equity
        .set_index("Date")[
            "Equity"
        ]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Return",
    ]

    # ========================================================
    # YEARLY RETURNS
    # ========================================================

    yearly = (
        equity
        .set_index("Date")[
            "Equity"
        ]
        .resample("YE")
        .last()
        .pct_change()
        .dropna()
        .reset_index()
    )

    yearly.columns = [
        "Year",
        "Return",
    ]

    # ========================================================
    # STOCK ATTRIBUTION
    # ========================================================

    if not trades_df.empty:

        attribution = (

            trades_df

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

    else:

        attribution = (
            pd.DataFrame()
        )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    if not trades_df.empty:

        exit_analysis = (

            trades_df

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

    else:

        exit_analysis = (
            pd.DataFrame()
        )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "Final_Equity":
            final_equity,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Max_Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Calmar":
            calmar,

        "Volatility":
            volatility,

        "Win_Rate":
            win_rate,

        "Profit_Factor":
            profit_factor,

        "Avg_Trade":
            avg_trade,

        "Best_Trade":
            best_trade,

        "Worst_Trade":
            worst_trade,

        "Trades":
            len(trades_df),

        "Equity":
            equity,

        "Trades_Data":
            trades_df,

        "Monthly":
            monthly,

        "Yearly":
            yearly,

        "Attribution":
            attribution,

        "Exit_Analysis":
            exit_analysis,

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
        "V30 - DAILY PORTFOLIO VALIDATION"
    )

    print("=" * 70)

    predictions = (
        load_predictions()
    )

    market_data = (
        load_market_data()
    )

    print(
        f"\nPredictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Prediction period: "
        f"{predictions['Date'].min().date()}"
        f" -> "
        f"{predictions['Date'].max().date()}"
    )

    result = run_backtest(
        predictions,
        market_data
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "V30 DAILY VALIDATION RESULTS"
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
        f"Daily Sharpe: "
        f"{result['Sharpe']:.2f}"
    )

    print(
        f"Sortino: "
        f"{result['Sortino']:.2f}"
    )

    print(
        f"Calmar: "
        f"{result['Calmar']:.2f}"
    )

    print(
        f"Annualized Volatility: "
        f"{result['Volatility'] * 100:.2f}%"
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
        f"Best Trade: "
        f"{result['Best_Trade'] * 100:.2f}%"
    )

    print(
        f"Worst Trade: "
        f"{result['Worst_Trade'] * 100:.2f}%"
    )

    print(
        f"Trades: "
        f"{result['Trades']}"
    )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    if not result[
        "Exit_Analysis"
    ].empty:

        exit_df = (
            result[
                "Exit_Analysis"
            ].copy()
        )

        exit_df[
            "Avg_Return"
        ] *= 100

        exit_df[
            "Win_Rate"
        ] *= 100

        print("\n")
        print("=" * 70)

        print(
            "EXIT ANALYSIS"
        )

        print("=" * 70)

        print(
            exit_df.to_string(
                index=False
            )
        )

        exit_df.to_csv(

            OUTPUT_DIR
            / "v30_exit_analysis.csv",

            index=False

        )

    # ========================================================
    # STOCK ATTRIBUTION
    # ========================================================

    if not result[
        "Attribution"
    ].empty:

        attribution = (
            result[
                "Attribution"
            ].copy()
        )

        attribution[
            "Avg_Return"
        ] *= 100

        attribution[
            "Win_Rate"
        ] *= 100

        print("\n")
        print("=" * 70)

        print(
            "STOCK ATTRIBUTION"
        )

        print("=" * 70)

        print(
            attribution.to_string(
                index=False
            )
        )

        attribution.to_csv(

            OUTPUT_DIR
            / "v30_stock_attribution.csv",

            index=False

        )

    # ========================================================
    # YEARLY RETURNS
    # ========================================================

    yearly = (
        result[
            "Yearly"
        ].copy()
    )

    if not yearly.empty:

        yearly[
            "Return"
        ] *= 100

        print("\n")
        print("=" * 70)

        print(
            "YEARLY RETURNS"
        )

        print("=" * 70)

        print(
            yearly.to_string(
                index=False
            )
        )

        yearly.to_csv(

            OUTPUT_DIR
            / "v30_yearly_returns.csv",

            index=False

        )

    # ========================================================
    # SAVE EQUITY
    # ========================================================

    result[
        "Equity"
    ].to_csv(

        OUTPUT_DIR
        / "v30_daily_equity.csv",

        index=False

    )

    result[
        "Trades_Data"
    ].to_csv(

        OUTPUT_DIR
        / "v30_trade_log.csv",

        index=False

    )

    result[
        "Monthly"
    ].to_csv(

        OUTPUT_DIR
        / "v30_monthly_returns.csv",

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

        "Sortino":
            result[
                "Sortino"
            ],

        "Calmar":
            result[
                "Calmar"
            ],

        "Volatility":
            result[
                "Volatility"
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
        / "v30_results.csv",

        index=False

    )

    # ========================================================
    # COMPLETE
    # ========================================================

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
        "\nV30 COMPLETE"
    )


if __name__ == "__main__":

    main()

