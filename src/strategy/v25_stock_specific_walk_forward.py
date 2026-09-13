from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

MARKET_FILE = Path(
    "data/raw/nifty50.csv"
)

OUTPUT_DIR = Path(
    "data/models/v25_stock_specific"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INITIAL_CAPITAL = 100_000.0

TOP_N = 3

HOLDING_DAYS = 20

MIN_TRAIN_DAYS = 500

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

ONE_WAY_COST = (
    TRANSACTION_COST
    + SLIPPAGE
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

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

    datasets = []

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    print(
        f"Feature files found: "
        f"{len(files)}"
    )

    for file in files:

        symbol = extract_symbol(
            file
        )

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

        # ----------------------------------------------------
        # Future stock return
        # ----------------------------------------------------

        df[
            "Stock_Return_20D"
        ] = (

            df["Close"].shift(-HOLDING_DAYS)
            / df["Close"]
            - 1
        ) * 100

        # ----------------------------------------------------
        # Excess return vs NIFTY
        # ----------------------------------------------------

        if (
            "NIFTY_Return_20D"
            not in df.columns
        ):

            print(
                f"Skipping {symbol}: "
                f"NIFTY_Return_20D missing"
            )

            continue

        df[
            "Excess_Return_20D"
        ] = (

            df["Stock_Return_20D"]
            - df["NIFTY_Return_20D"]
        )

        df["Symbol"] = symbol

        available_features = [
            f
            for f in FEATURES
            if f in df.columns
        ]

        required = [
            "Date",
            "Symbol",
            "Close",
            "Stock_Return_20D",
            "NIFTY_Return_20D",
            "Excess_Return_20D",
        ]

        required += available_features

        df = df[
            list(
                dict.fromkeys(
                    required
                )
            )
        ].copy()

        datasets.append(
            df
        )

        print(
            f"{symbol:<18}"
            f"{len(df):>5} rows"
        )

    if not datasets:

        raise RuntimeError(
            "No usable datasets found."
        )

    return pd.concat(
        datasets,
        ignore_index=True
    )


# ============================================================
# PREPARE
# ============================================================

def prepare_data(data):

    feature_columns = [
        f
        for f in FEATURES
        if f in data.columns
    ]

    print(
        f"\nFeatures available: "
        f"{len(feature_columns)}"
    )

    # Fill missing values using same-date
    # cross-sectional median.

    for feature in feature_columns:

        data[
            feature
        ] = (

            data
            .groupby("Date")[
                feature
            ]
            .transform(
                lambda x:
                x.fillna(
                    x.median()
                )
            )
        )

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    data = data.dropna(
        subset=[
            "Excess_Return_20D"
        ]
        + feature_columns
    )

    data = (
        data
        .sort_values(
            [
                "Date",
                "Symbol"
            ]
        )
        .reset_index(drop=True)
    )

    return (
        data,
        feature_columns
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    return RandomForestRegressor(

        n_estimators=400,

        max_depth=8,

        min_samples_split=15,

        min_samples_leaf=5,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1,
    )


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward(
    data,
    feature_columns
):

    all_predictions = []

    symbols = sorted(
        data["Symbol"].unique()
    )

    dates = sorted(
        data["Date"].unique()
    )

    # Same rebalance dates used by V24.

    rebalance_dates = dates[
        MIN_TRAIN_DAYS::HOLDING_DAYS
    ]

    print(
        f"\nStocks: "
        f"{len(symbols)}"
    )

    print(
        f"Rebalance dates: "
        f"{len(rebalance_dates)}"
    )

    print("\n")

    # --------------------------------------------------------
    # Each rebalance
    # --------------------------------------------------------

    for rebalance_number, test_date in enumerate(
        rebalance_dates,
        start=1
    ):

        cutoff_index = (
            dates.index(test_date)
            - HOLDING_DAYS
        )

        if cutoff_index <= 0:
            continue

        train_end_date = dates[
            cutoff_index - 1
        ]

        day_predictions = []

        # ----------------------------------------------------
        # Train ONE model per stock
        # ----------------------------------------------------

        for symbol in symbols:

            stock = data[
                data["Symbol"] == symbol
            ].copy()

            train = stock[
                stock["Date"]
                <= train_end_date
            ].copy()

            test = stock[
                stock["Date"]
                == test_date
            ].copy()

            if (
                len(train)
                < MIN_TRAIN_DAYS
            ):

                continue

            if test.empty:
                continue

            X_train = train[
                feature_columns
            ]

            y_train = train[
                "Excess_Return_20D"
            ]

            X_test = test[
                feature_columns
            ]

            model = build_model()

            model.fit(
                X_train,
                y_train
            )

            prediction = (
                model.predict(
                    X_test
                )[0]
            )

            row = test.iloc[0].copy()

            row[
                "Predicted_Excess_Return"
            ] = prediction

            row[
                "Rebalance_Number"
            ] = rebalance_number

            row[
                "Train_End_Date"
            ] = train_end_date

            row[
                "Test_Date"
            ] = test_date

            day_predictions.append(
                row
            )

        if day_predictions:

            day_result = pd.DataFrame(
                day_predictions
            )

            all_predictions.append(
                day_result
            )

            print(
                f"Rebalance "
                f"{rebalance_number:>2} | "
                f"{pd.Timestamp(test_date).date()} | "
                f"Train through "
                f"{pd.Timestamp(train_end_date).date()} | "
                f"Stocks: "
                f"{len(day_result)}"
            )

    if not all_predictions:

        raise RuntimeError(
            "No predictions generated."
        )

    return pd.concat(
        all_predictions,
        ignore_index=True
    )


# ============================================================
# PORTFOLIO
# ============================================================

def create_portfolio(
    predictions
):

    rows = []

    for (
        rebalance_number,
        day
    ) in predictions.groupby(
        "Rebalance_Number"
    ):

        if len(day) < TOP_N:
            continue

        selected = (
            day
            .sort_values(
                "Predicted_Excess_Return",
                ascending=False
            )
            .head(TOP_N)
        )

        for rank, (_, row) in enumerate(
            selected.iterrows(),
            start=1
        ):

            rows.append({

                "Rebalance_Number":
                    rebalance_number,

                "Entry_Date":
                    row["Test_Date"],

                "Symbol":
                    row["Symbol"],

                "Rank":
                    rank,

                "Predicted_Excess_Return":
                    row[
                        "Predicted_Excess_Return"
                    ],

                "Actual_Excess_Return":
                    row[
                        "Excess_Return_20D"
                    ],

                "Stock_Return_20D":
                    row[
                        "Stock_Return_20D"
                    ],
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    portfolio
):

    portfolio = portfolio.copy()

    portfolio[
        "Gross_Return"
    ] = (
        portfolio[
            "Stock_Return_20D"
        ] / 100
    )

    portfolio[
        "Net_Return"
    ] = (

        (
            1
            + portfolio[
                "Gross_Return"
            ]
        )

        * (1 - ONE_WAY_COST)

        * (1 - ONE_WAY_COST)

        - 1
    )

    portfolio[
        "Net_Return_Pct"
    ] = (
        portfolio[
            "Net_Return"
        ] * 100
    )

    rebalance_returns = (
        portfolio
        .groupby(
            "Entry_Date"
        )[
            "Net_Return"
        ]
        .mean()
        .sort_index()
    )

    equity = INITIAL_CAPITAL

    equity_rows = []

    for date, ret in (
        rebalance_returns.items()
    ):

        equity *= (
            1 + ret
        )

        equity_rows.append({

            "Date":
                date,

            "Portfolio_Return":
                ret,

            "Portfolio_Return_Pct":
                ret * 100,

            "Equity":
                equity,
        })

    equity_curve = pd.DataFrame(
        equity_rows
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_equity = (
        equity_curve[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    days = (
        equity_curve["Date"].iloc[-1]
        - equity_curve["Date"].iloc[0]
    ).days

    years = (
        days / 365.25
    )

    if years > 0:

        cagr = (
            (
                final_equity
                / INITIAL_CAPITAL
            )
            ** (1 / years)
            - 1
        )

    else:

        cagr = np.nan

    peak = (
        equity_curve[
            "Equity"
        ].cummax()
    )

    drawdown = (
        equity_curve[
            "Equity"
        ]
        / peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    if (
        len(rebalance_returns) > 1
        and rebalance_returns.std() > 0
    ):

        sharpe = (
            rebalance_returns.mean()
            / rebalance_returns.std()
            * np.sqrt(
                252 / HOLDING_DAYS
            )
        )

    else:

        sharpe = np.nan

    win_rate = (
        portfolio[
            "Net_Return"
        ] > 0
    ).mean()

    gross_profit = (
        portfolio.loc[
            portfolio["Net_Return"] > 0,
            "Net_Return"
        ].sum()
    )

    gross_loss = abs(
        portfolio.loc[
            portfolio["Net_Return"] < 0,
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

    average_trade = (
        portfolio[
            "Net_Return"
        ].mean()
    )

    metrics = {

        "Initial_Capital":
            INITIAL_CAPITAL,

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

        "Win_Rate":
            win_rate,

        "Profit_Factor":
            profit_factor,

        "Average_Trade":
            average_trade,

        "Trades":
            len(portfolio),

        "Rebalances":
            len(rebalance_returns),
    }

    return (
        portfolio,
        equity_curve,
        metrics
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
            "Test_Date"
        )
    ):

        if len(group) < 5:
            continue

        ic = (
            group[
                "Predicted_Excess_Return"
            ]
            .corr(
                group[
                    "Excess_Return_20D"
                ],
                method="spearman"
            )
        )

        if pd.notna(ic):

            values.append(ic)

    if not values:

        return (
            np.nan,
            np.nan
        )

    return (
        np.mean(values),
        np.median(values)
    )


# ============================================================
# STOCK ATTRIBUTION
# ============================================================

def stock_attribution(
    portfolio
):

    return (
        portfolio
        .groupby("Symbol")
        .agg(

            Trades=(
                "Net_Return",
                "count"
            ),

            Average_Return=(
                "Net_Return",
                "mean"
            ),

            Total_Return=(
                "Net_Return",
                "sum"
            ),

            Win_Rate=(
                "Net_Return",
                lambda x:
                (x > 0).mean()
            )
        )
        .sort_values(
            "Total_Return",
            ascending=False
        )
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_with_v24():

    v24_file = Path(
        "data/models/v24_walk_forward/"
        "v24_metrics.csv"
    )

    if not v24_file.exists():

        return None

    df = pd.read_csv(
        v24_file
    )

    return df.iloc[0].to_dict()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("V25 - STOCK-SPECIFIC WALK-FORWARD")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    data, feature_columns = (
        prepare_data(data)
    )

    # --------------------------------------------------------
    # Walk-forward
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STOCK-SPECIFIC WALK-FORWARD")
    print("=" * 70)

    predictions = walk_forward(
        data,
        feature_columns
    )

    print("\n")
    print(
        f"Total predictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Rebalances: "
        f"{predictions['Rebalance_Number'].nunique()}"
    )

    # --------------------------------------------------------
    # Rank IC
    # --------------------------------------------------------

    mean_ic, median_ic = (
        calculate_rank_ic(
            predictions
        )
    )

    print(
        f"Mean Rank IC: "
        f"{mean_ic:.4f}"
    )

    print(
        f"Median Rank IC: "
        f"{median_ic:.4f}"
    )

    # --------------------------------------------------------
    # Portfolio
    # --------------------------------------------------------

    portfolio = create_portfolio(
        predictions
    )

    print(
        f"Portfolio positions: "
        f"{len(portfolio):,}"
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    (
        portfolio,
        equity_curve,
        metrics
    ) = backtest(
        portfolio
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("V25 RESULTS")
    print("=" * 70)

    print(
        f"Initial Capital: "
        f"₹{metrics['Initial_Capital']:,.2f}"
    )

    print(
        f"Final Equity:    "
        f"₹{metrics['Final_Equity']:,.2f}"
    )

    print(
        f"Total Return:    "
        f"{metrics['Total_Return'] * 100:.2f}%"
    )

    print(
        f"CAGR:             "
        f"{metrics['CAGR'] * 100:.2f}%"
    )

    print(
        f"Max Drawdown:     "
        f"{metrics['Max_Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe:           "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Win Rate:         "
        f"{metrics['Win_Rate'] * 100:.2f}%"
    )

    print(
        f"Profit Factor:    "
        f"{metrics['Profit_Factor']:.2f}"
    )

    print(
        f"Average Trade:    "
        f"{metrics['Average_Trade'] * 100:.3f}%"
    )

    print(
        f"Trades:           "
        f"{metrics['Trades']}"
    )

    print(
        f"Rebalances:       "
        f"{metrics['Rebalances']}"
    )

    # --------------------------------------------------------
    # V24 comparison
    # --------------------------------------------------------

    v24 = compare_with_v24()

    if v24:

        print("\n")
        print("=" * 70)
        print("V24 vs V25")
        print("=" * 70)

        print(
            f"V24 Return: "
            f"{v24['Total_Return'] * 100:.2f}%"
        )

        print(
            f"V25 Return: "
            f"{metrics['Total_Return'] * 100:.2f}%"
        )

        print(
            f"V24 CAGR: "
            f"{v24['CAGR'] * 100:.2f}%"
        )

        print(
            f"V25 CAGR: "
            f"{metrics['CAGR'] * 100:.2f}%"
        )

        print(
            f"V24 Sharpe: "
            f"{v24['Sharpe']:.2f}"
        )

        print(
            f"V25 Sharpe: "
            f"{metrics['Sharpe']:.2f}"
        )

    # --------------------------------------------------------
    # Attribution
    # --------------------------------------------------------

    attribution = (
        stock_attribution(
            portfolio
        )
    )

    print("\n")
    print("=" * 70)
    print("STOCK ATTRIBUTION")
    print("=" * 70)

    print(
        attribution.to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    predictions_file = (
        OUTPUT_DIR
        / "v25_predictions.csv"
    )

    portfolio_file = (
        OUTPUT_DIR
        / "v25_trade_log.csv"
    )

    equity_file = (
        OUTPUT_DIR
        / "v25_equity_curve.csv"
    )

    attribution_file = (
        OUTPUT_DIR
        / "v25_stock_attribution.csv"
    )

    metrics_file = (
        OUTPUT_DIR
        / "v25_metrics.csv"
    )

    ranking_file = (
        OUTPUT_DIR
        / "v25_ranking_metrics.csv"
    )

    predictions.to_csv(
        predictions_file,
        index=False
    )

    portfolio.to_csv(
        portfolio_file,
        index=False
    )

    equity_curve.to_csv(
        equity_file,
        index=False
    )

    attribution.to_csv(
        attribution_file
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_file,
        index=False
    )

    pd.DataFrame(
        [{
            "Mean_Rank_IC":
                mean_ic,

            "Median_Rank_IC":
                median_ic,
        }]
    ).to_csv(
        ranking_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        predictions_file.resolve()
    )

    print(
        portfolio_file.resolve()
    )

    print(
        equity_file.resolve()
    )

    print(
        attribution_file.resolve()
    )

    print(
        metrics_file.resolve()
    )

    print(
        ranking_file.resolve()
    )

    print("\n")
    print("=" * 70)
    print("V25 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()