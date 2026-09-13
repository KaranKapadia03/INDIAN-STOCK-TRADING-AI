from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "walk_forward_portfolio"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INITIAL_CAPITAL = 100000.0

INITIAL_TRAIN_RATIO = 0.60

RETRAIN_EVERY = 60

HOLDING_PERIOD = 5

MAX_POSITIONS = 6

MAX_STOCK_EXPOSURE = 0.20

MAX_PORTFOLIO_EXPOSURE = 0.80

BUY_THRESHOLD = 0.60

FINAL_SCORE_THRESHOLD = 0.15

ML_WEIGHT = 0.50

TECHNICAL_WEIGHT = 0.30

NEWS_WEIGHT = 0.20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

ATR_STOP_MULTIPLIER = 2.0

RISK_REWARD_RATIO = 2.0

RANDOM_STATE = 42


# ============================================================
# STOCK UNIVERSE
# ============================================================

STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "ULTRACEMCO.NS": "UltraTech Cement",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_stock(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features_news.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing feature file: {path}"
        )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df.sort_values(
        "date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(row):

    score = 0

    rsi = row["RSI_14"]

    if rsi < 30:

        score += 2

    elif rsi < 40:

        score += 1

    elif rsi > 70:

        score -= 2

    elif rsi > 60:

        score -= 1


    if row["MACD"] > row["MACD_Signal"]:

        score += 1

    else:

        score -= 1


    if row["Close"] > row["SMA_20"]:

        score += 1

    else:

        score -= 1


    if row["SMA_20"] > row["SMA_50"]:

        score += 1

    else:

        score -= 1


    if row["Close"] <= row["BB_Lower"]:

        score += 1

    elif row["Close"] >= row["BB_Upper"]:

        score -= 1


    return np.clip(
        score / 6,
        -1,
        1
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    required = [
        "Close",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "SMA_20",
        "SMA_50",
        "BB_Lower",
        "BB_Upper",
    ]

    df = df.dropna(
        subset=required
    ).copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df):

    future_return = (
        df["Close"].shift(-HOLDING_PERIOD)
        / df["Close"]
        - 1
    )

    df["Future_Return_5D"] = (
        future_return * 100
    )

    df["Target"] = (
        df["Future_Return_5D"] > 0
    ).astype(int)

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(df):

    excluded = {
        "date",
        "symbol",
        "Target",
        "Future_Return_5D",
    }

    candidates = []

    for column in df.columns:

        if column in excluded:
            continue

        if (
            pd.api.types
            .is_numeric_dtype(df[column])
        ):

            candidates.append(column)

    # Remove news columns from the ML model.
    #
    # The original models were trained using the
    # market/technical feature set.

    candidates = [
        column
        for column in candidates
        if not column.startswith("News_")
        and column != "Days_Since_News"
    ]

    return candidates


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_df, features):

    X = train_df[features]

    y = train_df["Target"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# CALCULATE POSITION SIZE
# ============================================================

def calculate_position_size(
    probability,
    final_score,
    volatility,
):

    if probability < 0.50:

        return 0.0

    confidence_score = (
        probability - 0.50
    ) / 0.40

    confidence_score = np.clip(
        confidence_score,
        0,
        1
    )

    signal_score = np.clip(
        abs(final_score),
        0,
        1
    )

    volatility_factor = (
        1
        / (
            1
            + max(volatility, 0)
        )
    )

    conviction = (
        confidence_score * 0.60
        +
        signal_score * 0.40
    )

    position = (
        MAX_STOCK_EXPOSURE
        * conviction
        * volatility_factor
        * 1.5
    )

    return float(
        min(
            position,
            MAX_STOCK_EXPOSURE
        )
    )


# ============================================================
# PREPARE WALK-FORWARD PREDICTIONS
# ============================================================

def generate_predictions(symbol, df):

    df = create_target(df)

    features = get_model_features(
        df
    )

    # Remove rows that cannot be used for training.

    df = df.dropna(
        subset=features + ["Target"]
    ).copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    total_rows = len(df)

    initial_train_size = int(
        total_rows
        * INITIAL_TRAIN_RATIO
    )

    predictions = []

    train_end = initial_train_size

    last_train_end = -1

    while train_end < total_rows:

        # ----------------------------------------------------
        # Train only on the past
        # ----------------------------------------------------

        train_df = df.iloc[
            :train_end
        ].copy()

        # We cannot use the final rows whose target
        # depends on future data beyond the dataset.

        train_df = train_df.dropna(
            subset=["Future_Return_5D"]
        )

        if (
            train_df["Target"]
            .nunique()
            < 2
        ):

            break

        model = train_model(
            train_df,
            features
        )

        # ----------------------------------------------------
        # Predict the next walk-forward window
        # ----------------------------------------------------

        test_end = min(
            train_end + RETRAIN_EVERY,
            total_rows
        )

        test_df = df.iloc[
            train_end:test_end
        ].copy()

        if test_df.empty:

            break

        probabilities = (
            model.predict_proba(
                test_df[features]
            )[:, 1]
        )

        test_df[
            "ML_Probability"
        ] = probabilities

        test_df[
            "ML_Score"
        ] = (
            test_df["ML_Probability"]
            * 2
            - 1
        )

        # ----------------------------------------------------
        # Technical score
        # ----------------------------------------------------

        test_df[
            "Technical_Score"
        ] = test_df.apply(
            technical_score,
            axis=1
        )

        # ----------------------------------------------------
        # Historical news
        # ----------------------------------------------------

        if "News_Sentiment" in test_df.columns:

            test_df[
                "News_Score"
            ] = test_df[
                "News_Sentiment"
            ].fillna(0)

        else:

            test_df[
                "News_Score"
            ] = 0.0

        if "News_Available" in test_df.columns:

            test_df[
                "News_Score"
            ] = np.where(
                test_df[
                    "News_Available"
                ] == 1,
                test_df[
                    "News_Score"
                ],
                0
            )

        # ----------------------------------------------------
        # Final signal
        # ----------------------------------------------------

        test_df[
            "Final_Score"
        ] = (
            test_df["ML_Score"]
            * ML_WEIGHT
            +
            test_df["Technical_Score"]
            * TECHNICAL_WEIGHT
            +
            test_df["News_Score"]
            * NEWS_WEIGHT
        )

        test_df[
            "Signal"
        ] = np.where(
            (
                test_df[
                    "ML_Probability"
                ]
                >= BUY_THRESHOLD
            )
            &
            (
                test_df[
                    "Final_Score"
                ]
                > FINAL_SCORE_THRESHOLD
            ),
            "BUY",
            "HOLD"
        )

        test_df[
            "symbol"
        ] = symbol

        predictions.append(
            test_df
        )

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        train_end = test_end

        if train_end == last_train_end:

            break

        last_train_end = train_end

    if not predictions:

        return pd.DataFrame()

    return pd.concat(
        predictions,
        ignore_index=True
    )


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def run_portfolio_backtest(
    predictions
):

    all_dates = sorted(
        predictions[
            "date"
        ].unique()
    )

    cash = INITIAL_CAPITAL

    positions = []

    trades = []

    equity_records = []

    for current_date in all_dates:

        # ====================================================
        # EXIT POSITIONS
        # ====================================================

        remaining_positions = []

        for position in positions:

            symbol = position[
                "symbol"
            ]

            rows = predictions[
                (
                    predictions["symbol"]
                    == symbol
                )
                &
                (
                    predictions["date"]
                    == current_date
                )
            ]

            if rows.empty:

                remaining_positions.append(
                    position
                )

                continue

            row = rows.iloc[0]

            close = float(
                row["Close"]
            )

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            exit_reason = None

            exit_price = None

            # ------------------------------------------------
            # ATR stop
            # ------------------------------------------------

            atr_percent = (
                position["atr_percent"]
            )

            stop_loss = (
                position["entry_price"]
                * (
                    1
                    - ATR_STOP_MULTIPLIER
                    * atr_percent
                )
            )

            take_profit = (
                position["entry_price"]
                * (
                    1
                    + ATR_STOP_MULTIPLIER
                    * atr_percent
                    * RISK_REWARD_RATIO
                )
            )

            if low <= stop_loss:

                exit_price = stop_loss

                exit_reason = (
                    "STOP_LOSS"
                )

            elif high >= take_profit:

                exit_price = take_profit

                exit_reason = (
                    "TAKE_PROFIT"
                )

            elif (
                current_date
                >= position[
                    "planned_exit_date"
                ]
            ):

                exit_price = close

                exit_reason = (
                    "TIME_EXIT"
                )

            else:

                remaining_positions.append(
                    position
                )

                continue

            # ------------------------------------------------
            # Exit slippage
            # ------------------------------------------------

            exit_price *= (
                1
                - SLIPPAGE
            )

            allocation = position[
                "allocation"
            ]

            gross_return = (
                exit_price
                / position["entry_price"]
                - 1
            )

            gross_profit = (
                allocation
                * gross_return
            )

            exit_cost = (
                allocation
                * TRANSACTION_COST
            )

            net_profit = (
                gross_profit
                - position["entry_cost"]
                - exit_cost
            )

            cash += (
                allocation
                + gross_profit
                - exit_cost
            )

            trades.append(
                {
                    "symbol":
                        symbol,
                    "entry_date":
                        position[
                            "entry_date"
                        ],
                    "exit_date":
                        current_date,
                    "entry_price":
                        position[
                            "entry_price"
                        ],
                    "exit_price":
                        exit_price,
                    "allocation":
                        allocation,
                    "return":
                        gross_return,
                    "net_profit":
                        net_profit,
                    "exit_reason":
                        exit_reason,
                    "ml_probability":
                        position[
                            "ml_probability"
                        ],
                    "technical_score":
                        position[
                            "technical_score"
                        ],
                    "news_score":
                        position[
                            "news_score"
                        ],
                    "final_score":
                        position[
                            "final_score"
                        ],
                }
            )

        positions = remaining_positions

        # ====================================================
        # FIND BUY CANDIDATES
        # ====================================================

        day_data = predictions[
            predictions["date"]
            == current_date
        ].copy()

        day_data = day_data[
            day_data["Signal"]
            == "BUY"
        ]

        if not day_data.empty:

            # Remove already-held stocks.

            held_symbols = {
                position[
                    "symbol"
                ]
                for position in positions
            }

            day_data = day_data[
                ~day_data[
                    "symbol"
                ].isin(
                    held_symbols
                )
            ]

            # Rank strongest signals.

            day_data.sort_values(
                "Final_Score",
                ascending=False,
                inplace=True
            )

            available_slots = (
                MAX_POSITIONS
                - len(positions)
            )

            day_data = day_data.head(
                max(
                    available_slots,
                    0
                )
            )

            current_exposure = sum(
                position[
                    "allocation"
                ]
                for position in positions
            )

            max_exposure = (
                INITIAL_CAPITAL
                * MAX_PORTFOLIO_EXPOSURE
            )

            remaining_capacity = (
                max_exposure
                - current_exposure
            )

            # ------------------------------------------------
            # Buy candidates
            # ------------------------------------------------

            for _, row in day_data.iterrows():

                if (
                    remaining_capacity
                    <= 0
                ):

                    break

                if cash <= 0:

                    break

                # Recent volatility.

                symbol = row[
                    "symbol"
                ]

                stock = predictions[
                    predictions[
                        "symbol"
                    ] == symbol
                ]

                historical = stock[
                    stock["date"]
                    <= current_date
                ].tail(20)

                volatility = (
                    historical[
                        "Close"
                    ]
                    .pct_change()
                    .std()
                    * np.sqrt(252)
                )

                if pd.isna(volatility):

                    volatility = 0

                position_percent = (
                    calculate_position_size(
                        probability=
                            float(
                                row[
                                    "ML_Probability"
                                ]
                            ),
                        final_score=
                            float(
                                row[
                                    "Final_Score"
                                ]
                            ),
                        volatility=
                            volatility
                    )
                )

                allocation = (
                    INITIAL_CAPITAL
                    * position_percent
                )

                allocation = min(
                    allocation,
                    remaining_capacity,
                    cash
                )

                if allocation <= 0:

                    continue

                # ------------------------------------------------
                # Entry
                # ------------------------------------------------

                entry_price = (
                    float(
                        row["Close"]
                    )
                    * (
                        1
                        + SLIPPAGE
                    )
                )

                entry_cost = (
                    allocation
                    * TRANSACTION_COST
                )

                if (
                    allocation
                    + entry_cost
                    > cash
                ):

                    continue

                cash -= (
                    allocation
                    + entry_cost
                )

                remaining_capacity -= (
                    allocation
                )

                # ------------------------------------------------
                # Planned exit
                # ------------------------------------------------

                future_dates = (
                    stock[
                        stock["date"]
                        > current_date
                    ]["date"]
                    .tolist()
                )

                if len(
                    future_dates
                ) >= HOLDING_PERIOD:

                    planned_exit_date = (
                        future_dates[
                            HOLDING_PERIOD - 1
                        ]
                    )

                elif future_dates:

                    planned_exit_date = (
                        future_dates[-1]
                    )

                else:

                    planned_exit_date = (
                        current_date
                    )

                positions.append(
                    {
                        "symbol":
                            symbol,
                        "entry_date":
                            current_date,
                        "planned_exit_date":
                            planned_exit_date,
                        "entry_price":
                            entry_price,
                        "allocation":
                            allocation,
                        "entry_cost":
                            entry_cost,
                        "ml_probability":
                            float(
                                row[
                                    "ML_Probability"
                                ]
                            ),
                        "technical_score":
                            float(
                                row[
                                    "Technical_Score"
                                ]
                            ),
                        "news_score":
                            float(
                                row[
                                    "News_Score"
                                ]
                            ),
                        "final_score":
                            float(
                                row[
                                    "Final_Score"
                                ]
                            ),
                        "atr_percent":
                            float(
                                row[
                                    "ATR_Percent"
                                ]
                            ),
                    }
                )

        # ====================================================
        # CALCULATE DAILY EQUITY
        # ====================================================

        portfolio_value = cash

        for position in positions:

            symbol = position[
                "symbol"
            ]

            rows = predictions[
                (
                    predictions["symbol"]
                    == symbol
                )
                &
                (
                    predictions["date"]
                    == current_date
                )
            ]

            if rows.empty:

                continue

            current_price = float(
                rows.iloc[0]["Close"]
            )

            position_value = (
                position[
                    "allocation"
                ]
                * (
                    current_price
                    / position[
                        "entry_price"
                    ]
                )
            )

            portfolio_value += (
                position_value
            )

        equity_records.append(
            {
                "date":
                    current_date,
                "portfolio_value":
                    portfolio_value,
                "cash":
                    cash,
                "positions":
                    len(positions),
            }
        )

    # ========================================================
    # CLOSE REMAINING POSITIONS
    # ========================================================

    for position in positions:

        symbol = position[
            "symbol"
        ]

        stock = predictions[
            predictions[
                "symbol"
            ] == symbol
        ]

        if stock.empty:

            continue

        row = stock.iloc[-1]

        exit_price = (
            float(
                row["Close"]
            )
            * (
                1
                - SLIPPAGE
            )
        )

        allocation = position[
            "allocation"
        ]

        gross_return = (
            exit_price
            / position[
                "entry_price"
            ]
            - 1
        )

        gross_profit = (
            allocation
            * gross_return
        )

        exit_cost = (
            allocation
            * TRANSACTION_COST
        )

        net_profit = (
            gross_profit
            - position[
                "entry_cost"
            ]
            - exit_cost
        )

        trades.append(
            {
                "symbol":
                    symbol,
                "entry_date":
                    position[
                        "entry_date"
                    ],
                "exit_date":
                    row["date"],
                "entry_price":
                    position[
                        "entry_price"
                    ],
                "exit_price":
                    exit_price,
                "allocation":
                    allocation,
                "return":
                    gross_return,
                "net_profit":
                    net_profit,
                "exit_reason":
                    "END_OF_DATA",
                "ml_probability":
                    position[
                        "ml_probability"
                    ],
                "technical_score":
                    position[
                        "technical_score"
                    ],
                "news_score":
                    position[
                        "news_score"
                    ],
                "final_score":
                    position[
                        "final_score"
                    ],
            }
        )

    return (
        pd.DataFrame(equity_records),
        pd.DataFrame(trades)
    )


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_metrics(
    equity,
    trades
):

    if equity.empty:

        return {}

    equity = equity.copy()

    equity[
        "peak"
    ] = equity[
        "portfolio_value"
    ].cummax()

    equity[
        "drawdown"
    ] = (
        equity[
            "portfolio_value"
        ]
        / equity["peak"]
        - 1
    )

    final_capital = float(
        equity[
            "portfolio_value"
        ].iloc[-1]
    )

    total_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    )

    max_drawdown = float(
        equity[
            "drawdown"
        ].min()
    )

    daily_returns = (
        equity[
            "portfolio_value"
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

        sharpe = 0.0

    if trades.empty:

        win_rate = 0.0

        average_trade = 0.0

        profit_factor = 0.0

    else:

        win_rate = (
            trades["return"] > 0
        ).mean()

        average_trade = (
            trades["return"].mean()
        )

        profits = trades.loc[
            trades[
                "net_profit"
            ] > 0,
            "net_profit"
        ].sum()

        losses = abs(
            trades.loc[
                trades[
                    "net_profit"
                ] < 0,
                "net_profit"
            ].sum()
        )

        if losses > 0:

            profit_factor = (
                profits / losses
            )

        else:

            profit_factor = np.inf

    return {
        "initial_capital":
            INITIAL_CAPITAL,

        "final_capital":
            final_capital,

        "return_pct":
            total_return * 100,

        "trades":
            len(trades),

        "win_rate":
            win_rate * 100,

        "average_trade_pct":
            average_trade * 100,

        "max_drawdown_pct":
            max_drawdown * 100,

        "sharpe":
            sharpe,

        "profit_factor":
            profit_factor,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "WALK-FORWARD PORTFOLIO BACKTEST"
    )
    print("=" * 80)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Models are retrained using only historical data "
        "available before each prediction window."
    )

    print(
        "\nInitial train ratio:",
        INITIAL_TRAIN_RATIO
    )

    print(
        "Retraining frequency:",
        RETRAIN_EVERY,
        "days"
    )

    print(
        "Holding period:",
        HOLDING_PERIOD,
        "days"
    )

    # --------------------------------------------------------
    # Load all stocks
    # --------------------------------------------------------

    all_predictions = []

    print(
        "\nGenerating walk-forward predictions..."
    )

    for symbol in STOCKS:

        try:

            df = load_stock(
                symbol
            )

            df = prepare_features(
                df
            )

            predictions = (
                generate_predictions(
                    symbol,
                    df
                )
            )

            if predictions.empty:

                print(
                    f"NO PREDICTIONS: {symbol}"
                )

                continue

            all_predictions.append(
                predictions
            )

            print(
                f"OK: {symbol} | "
                f"{len(predictions)} predictions"
            )

        except Exception as e:

            print(
                f"ERROR: {symbol} -> {e}"
            )

    if not all_predictions:

        print(
            "\nNo predictions generated."
        )

        return

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions.sort_values(
        "date",
        inplace=True
    )

    predictions.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_file = (
        OUTPUT_DIR
        / "walk_forward_predictions.csv"
    )

    predictions.to_csv(
        predictions_file,
        index=False
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        predictions_file
    )

    # --------------------------------------------------------
    # Run portfolio
    # --------------------------------------------------------

    equity, trades = (
        run_portfolio_backtest(
            predictions
        )
    )

    metrics = calculate_metrics(
        equity,
        trades
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    equity_file = (
        OUTPUT_DIR
        / "portfolio_equity_curve.csv"
    )

    trades_file = (
        OUTPUT_DIR
        / "portfolio_trades.csv"
    )

    performance_file = (
        OUTPUT_DIR
        / "portfolio_performance.csv"
    )

    equity.to_csv(
        equity_file,
        index=False
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        performance_file,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "WALK-FORWARD RESULTS"
    )
    print("=" * 80)

    print(
        f"\nInitial capital: "
        f"₹{metrics['initial_capital']:,.2f}"
    )

    print(
        f"Final capital: "
        f"₹{metrics['final_capital']:,.2f}"
    )

    print(
        f"Total return: "
        f"{metrics['return_pct']:.2f}%"
    )

    print(
        f"Total trades: "
        f"{metrics['trades']}"
    )

    print(
        f"Win rate: "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"Average trade: "
        f"{metrics['average_trade_pct']:.2f}%"
    )

    print(
        f"Maximum drawdown: "
        f"{metrics['max_drawdown_pct']:.2f}%"
    )

    print(
        f"Sharpe ratio: "
        f"{metrics['sharpe']:.2f}"
    )

    print(
        f"Profit factor: "
        f"{metrics['profit_factor']:.2f}"
    )

    # --------------------------------------------------------
    # Exit reasons
    # --------------------------------------------------------

    if not trades.empty:

        print("\n")
        print("=" * 80)
        print(
            "EXIT REASONS"
        )
        print("=" * 80)

        print(
            trades[
                "exit_reason"
            ]
            .value_counts()
            .to_string()
        )

    print("\n")
    print("=" * 80)
    print(
        "WALK-FORWARD BACKTEST COMPLETE"
    )
    print("=" * 80)

    print(
        "\nFiles saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()