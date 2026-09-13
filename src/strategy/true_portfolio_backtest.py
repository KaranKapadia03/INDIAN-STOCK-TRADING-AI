from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

OUTPUT_DIR = (
    MODEL_DIR / "true_portfolio_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000.0

ML_WEIGHT = 0.50
TECHNICAL_WEIGHT = 0.30
NEWS_WEIGHT = 0.20

BUY_ML_THRESHOLD = 0.60
BUY_FINAL_THRESHOLD = 0.15

MAX_STOCK_EXPOSURE = 0.20
MAX_PORTFOLIO_EXPOSURE = 0.80

MAX_POSITIONS = 6

HOLDING_PERIOD = 5

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

ATR_STOP_MULTIPLIER = 2.0
RISK_REWARD_RATIO = 2.0


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
# FEATURES
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Volatility_20D",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "BB_Position",
    "ATR_14",
    "ATR_Percent",
    "Volume_SMA_20",
    "Volume_Ratio",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",

    "NIFTY_Close",
    "NIFTY_Return_1D",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "NIFTY_SMA_20",
    "NIFTY_SMA_50",
    "NIFTY_Volatility_20D",
    "NIFTY_Price_vs_SMA20",
    "NIFTY_Price_vs_SMA50",

    "BANKNIFTY_Close",
    "BANKNIFTY_Return_1D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
    "BANKNIFTY_SMA_20",
    "BANKNIFTY_SMA_50",
    "BANKNIFTY_Volatility_20D",
    "BANKNIFTY_Price_vs_SMA20",
    "BANKNIFTY_Price_vs_SMA50",

    "Relative_5D_vs_NIFTY",
    "Relative_20D_vs_NIFTY",
    "Relative_5D_vs_BANKNIFTY",
    "Relative_20D_vs_BANKNIFTY",
    "Relative_Volatility_NIFTY",
    "Relative_Volatility_BANKNIFTY",
    "Relative_Price_vs_NIFTY",
    "Relative_Price_vs_BANKNIFTY",
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(row):

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

    return float(
        np.clip(
            score / 6,
            -1,
            1
        )
    )


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_stock(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features_news.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    data = pd.read_csv(
        path
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data.sort_values(
        "date",
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Keep rows with enough model information
    # --------------------------------------------------------

    available_features = [
        column
        for column in FEATURES
        if column in data.columns
    ]

    data.dropna(
        subset=available_features,
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(symbol):

    model_path = (
        MODEL_DIR
        / symbol.replace(".", "_")
        / "model.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


# ============================================================
# PREPARE PREDICTIONS
# ============================================================

def prepare_stock(symbol):

    data = load_stock(
        symbol
    )

    model = load_model(
        symbol
    )

    # --------------------------------------------------------
    # Use model's own feature list when available
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        model_features = [
            column
            for column in model.feature_names_in_
            if column in data.columns
        ]

    else:

        model_features = [
            column
            for column in FEATURES
            if column in data.columns
        ]

    # --------------------------------------------------------
    # ML probability
    # --------------------------------------------------------

    data[
        "ML_Probability"
    ] = model.predict_proba(
        data[model_features]
    )[:, 1]

    # --------------------------------------------------------
    # Technical score
    # --------------------------------------------------------

    data[
        "Technical_Score"
    ] = data.apply(
        calculate_technical_score,
        axis=1
    )

    # --------------------------------------------------------
    # ML score
    # --------------------------------------------------------

    data[
        "ML_Score"
    ] = (
        data["ML_Probability"]
        * 2
        - 1
    )

    # --------------------------------------------------------
    # Historical news signal
    #
    # We use ONLY information already present
    # on the historical date.
    # --------------------------------------------------------

    data[
        "News_Score"
    ] = data[
        "News_Sentiment"
    ].fillna(
        0
    )

    # Reduce influence when news is stale/unavailable.

    if "News_Available" in data.columns:

        data[
            "News_Score"
        ] = np.where(
            data["News_Available"] == 1,
            data["News_Score"],
            0
        )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    data[
        "Final_Score"
    ] = (
        data["ML_Score"]
        * ML_WEIGHT
        +
        data["Technical_Score"]
        * TECHNICAL_WEIGHT
        +
        data["News_Score"]
        * NEWS_WEIGHT
    )

    # --------------------------------------------------------
    # BUY signal
    # --------------------------------------------------------

    data[
        "Signal"
    ] = np.where(
        (
            data["ML_Probability"]
            >= BUY_ML_THRESHOLD
        )
        &
        (
            data["Final_Score"]
            > BUY_FINAL_THRESHOLD
        ),
        "BUY",
        "HOLD"
    )

    data[
        "symbol"
    ] = symbol

    return data


# ============================================================
# PREPARE ALL STOCKS
# ============================================================

def prepare_all_stocks():

    stocks = {}

    print(
        "\nPreparing historical stock data..."
    )

    for symbol in STOCKS:

        try:

            stocks[symbol] = prepare_stock(
                symbol
            )

            print(
                f"OK: {symbol}"
            )

        except Exception as e:

            print(
                f"ERROR: {symbol} -> {e}"
            )

    return stocks


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def run_backtest(stocks):

    # --------------------------------------------------------
    # Create common trading calendar
    # --------------------------------------------------------

    all_dates = sorted(
        set(
            date
            for data in stocks.values()
            for date in data["date"]
        )
    )

    cash = INITIAL_CAPITAL

    positions = []

    trades = []

    equity_records = []

    portfolio_value = INITIAL_CAPITAL

    # --------------------------------------------------------
    # Daily simulation
    # --------------------------------------------------------

    for current_date in all_dates:

        # ====================================================
        # EXIT EXISTING POSITIONS
        # ====================================================

        remaining_positions = []

        for position in positions:

            symbol = position["symbol"]

            data = stocks[symbol]

            rows = data[
                data["date"] == current_date
            ]

            if rows.empty:

                remaining_positions.append(
                    position
                )

                continue

            row = rows.iloc[0]

            current_price = float(
                row["Close"]
            )

            exit_reason = None

            exit_price = None

            # ------------------------------------------------
            # ATR stop / target
            # ------------------------------------------------

            atr_percent = float(
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

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            # Stop has priority if both
            # are touched on the same day.

            if low <= stop_loss:

                exit_price = stop_loss
                exit_reason = "STOP_LOSS"

            elif high >= take_profit:

                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"

            # ------------------------------------------------
            # Time-based exit
            # ------------------------------------------------

            elif (
                current_date
                >= position["planned_exit_date"]
            ):

                exit_price = current_price
                exit_reason = "TIME_EXIT"

            # ------------------------------------------------
            # Still holding
            # ------------------------------------------------

            else:

                remaining_positions.append(
                    position
                )

                continue

            # ------------------------------------------------
            # Apply exit slippage
            # ------------------------------------------------

            exit_price *= (
                1 - SLIPPAGE
            )

            allocation = position[
                "allocation"
            ]

            gross_profit = (
                allocation
                * (
                    exit_price
                    / position["entry_price"]
                    - 1
                )
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
                    "symbol": symbol,
                    "entry_date":
                        position["entry_date"],
                    "exit_date":
                        current_date,
                    "entry_price":
                        position["entry_price"],
                    "exit_price":
                        exit_price,
                    "allocation":
                        allocation,
                    "return":
                        (
                            exit_price
                            / position["entry_price"]
                            - 1
                        ),
                    "net_profit":
                        net_profit,
                    "exit_reason":
                        exit_reason,
                    "ml_probability":
                        position["ml_probability"],
                    "technical_score":
                        position["technical_score"],
                    "news_score":
                        position["news_score"],
                    "final_score":
                        position["final_score"],
                }
            )

        positions = remaining_positions

        # ====================================================
        # NEW BUY SIGNALS
        # ====================================================

        candidates = []

        for symbol, data in stocks.items():

            rows = data[
                data["date"] == current_date
            ]

            if rows.empty:
                continue

            row = rows.iloc[0]

            if row["Signal"] != "BUY":
                continue

            # Don't buy an already-held stock.

            if any(
                p["symbol"] == symbol
                for p in positions
            ):
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "ml_probability":
                        float(
                            row["ML_Probability"]
                        ),
                    "technical_score":
                        float(
                            row["Technical_Score"]
                        ),
                    "news_score":
                        float(
                            row["News_Score"]
                        ),
                    "final_score":
                        float(
                            row["Final_Score"]
                        ),
                    "price":
                        float(
                            row["Close"]
                        ),
                    "atr_percent":
                        float(
                            row["ATR_Percent"]
                        ),
                }
            )

        # Strongest signals first.

        candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        # Available portfolio capacity.

        current_exposure = sum(
            p["allocation"]
            for p in positions
        )

        available_capacity = (
            cash
            if cash > 0
            else 0
        )

        max_exposure_amount = (
            INITIAL_CAPITAL
            * MAX_PORTFOLIO_EXPOSURE
        )

        remaining_portfolio_capacity = (
            max_exposure_amount
            - current_exposure
        )

        slots = (
            MAX_POSITIONS
            - len(positions)
        )

        selected = candidates[
            :max(slots, 0)
        ]

        if selected:

            # ------------------------------------------------
            # Calculate allocations
            # ------------------------------------------------

            for candidate in selected:

                if remaining_portfolio_capacity <= 0:
                    break

                # --------------------------------------------
                # Approximate risk-engine sizing
                # --------------------------------------------

                confidence = (
                    candidate[
                        "ml_probability"
                    ]
                )

                confidence_score = (
                    confidence - 0.50
                ) / 0.40

                confidence_score = np.clip(
                    confidence_score,
                    0,
                    1
                )

                signal_score = np.clip(
                    abs(
                        candidate[
                            "final_score"
                        ]
                    ),
                    0,
                    1
                )

                # Estimate annualized volatility
                # from recent stock data.

                stock_data = stocks[
                    candidate["symbol"]
                ]

                recent = stock_data[
                    stock_data["date"]
                    <= current_date
                ].tail(20)

                daily_returns = (
                    recent["Close"]
                    .pct_change()
                )

                volatility = (
                    daily_returns.std()
                    * np.sqrt(252)
                )

                if pd.isna(volatility):
                    volatility = 0

                volatility_factor = (
                    1
                    / (
                        1
                        + volatility
                    )
                )

                conviction = (
                    confidence_score * 0.60
                    +
                    signal_score * 0.40
                )

                position_percent = (
                    MAX_STOCK_EXPOSURE
                    * conviction
                    * volatility_factor
                    * 1.5
                )

                position_percent = min(
                    position_percent,
                    MAX_STOCK_EXPOSURE
                )

                allocation = (
                    INITIAL_CAPITAL
                    * position_percent
                )

                allocation = min(
                    allocation,
                    remaining_portfolio_capacity,
                    cash
                )

                if allocation <= 0:
                    continue

                # --------------------------------------------
                # Entry
                # --------------------------------------------

                entry_price = (
                    candidate["price"]
                    * (
                        1
                        + SLIPPAGE
                    )
                )

                entry_cost = (
                    allocation
                    * TRANSACTION_COST
                )

                cash -= (
                    allocation
                    + entry_cost
                )

                remaining_portfolio_capacity -= (
                    allocation
                )

                # --------------------------------------------
                # Planned exit
                # --------------------------------------------

                stock_data = stocks[
                    candidate["symbol"]
                ]

                future_dates = (
                    stock_data[
                        stock_data["date"]
                        > current_date
                    ]["date"]
                    .tolist()
                )

                if len(future_dates) >= HOLDING_PERIOD:

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

                    planned_exit_date = current_date

                positions.append(
                    {
                        "symbol":
                            candidate["symbol"],
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
                            candidate[
                                "ml_probability"
                            ],
                        "technical_score":
                            candidate[
                                "technical_score"
                            ],
                        "news_score":
                            candidate[
                                "news_score"
                            ],
                        "final_score":
                            candidate[
                                "final_score"
                            ],
                        "atr_percent":
                            candidate[
                                "atr_percent"
                            ],
                    }
                )

        # ====================================================
        # DAILY PORTFOLIO VALUE
        # ====================================================

        portfolio_value = cash

        for position in positions:

            symbol = position["symbol"]

            data = stocks[symbol]

            rows = data[
                data["date"] == current_date
            ]

            if rows.empty:

                continue

            current_price = float(
                rows.iloc[0]["Close"]
            )

            position_value = (
                position["allocation"]
                * (
                    current_price
                    / position["entry_price"]
                )
            )

            portfolio_value += (
                position_value
            )

        equity_records.append(
            {
                "date": current_date,
                "portfolio_value":
                    portfolio_value,
                "cash": cash,
                "positions":
                    len(positions),
            }
        )

    # ========================================================
    # CLOSE REMAINING POSITIONS
    # ========================================================

    for position in positions:

        symbol = position["symbol"]

        data = stocks[symbol]

        last_row = data.iloc[-1]

        exit_price = float(
            last_row["Close"]
        )

        exit_price *= (
            1 - SLIPPAGE
        )

        allocation = position[
            "allocation"
        ]

        gross_profit = (
            allocation
            * (
                exit_price
                / position["entry_price"]
                - 1
            )
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

        trades.append(
            {
                "symbol":
                    symbol,
                "entry_date":
                    position["entry_date"],
                "exit_date":
                    last_row["date"],
                "entry_price":
                    position["entry_price"],
                "exit_price":
                    exit_price,
                "allocation":
                    allocation,
                "return":
                    (
                        exit_price
                        / position["entry_price"]
                        - 1
                    ),
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
# PERFORMANCE METRICS
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
        equity["portfolio_value"]
        / equity["peak"]
        - 1
    )

    final_value = float(
        equity[
            "portfolio_value"
        ].iloc[-1]
    )

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1
    )

    max_drawdown = float(
        equity["drawdown"].min()
    )

    daily_returns = (
        equity[
            "portfolio_value"
        ].pct_change()
        .dropna()
    )

    if (
        len(daily_returns) > 1
        and daily_returns.std() != 0
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
        profit_factor = 0.0
        average_trade = 0.0

    else:

        win_rate = (
            trades["return"] > 0
        ).mean()

        average_trade = (
            trades["return"].mean()
        )

        profits = trades.loc[
            trades["net_profit"] > 0,
            "net_profit"
        ].sum()

        losses = abs(
            trades.loc[
                trades["net_profit"] < 0,
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
            final_value,

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
    print("INDIAN STOCK TRADING AI")
    print("TRUE PORTFOLIO BACKTEST")
    print("=" * 80)

    print(
        f"\nInitial capital: "
        f"₹{INITIAL_CAPITAL:,.0f}"
    )

    print(
        f"Maximum stock exposure: "
        f"{MAX_STOCK_EXPOSURE * 100:.0f}%"
    )

    print(
        f"Maximum portfolio exposure: "
        f"{MAX_PORTFOLIO_EXPOSURE * 100:.0f}%"
    )

    print(
        f"Maximum positions: "
        f"{MAX_POSITIONS}"
    )

    print(
        f"Holding period: "
        f"{HOLDING_PERIOD} days"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.2f}%"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    stocks = prepare_all_stocks()

    print(
        f"\nStocks loaded: "
        f"{len(stocks)} / {len(STOCKS)}"
    )

    if not stocks:

        print(
            "No stocks available."
        )

        return

    # --------------------------------------------------------
    # Run backtest
    # --------------------------------------------------------

    equity, trades = run_backtest(
        stocks
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        equity,
        trades
    )

    # --------------------------------------------------------
    # Save
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
    print("BACKTEST RESULTS")
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
        print("EXIT REASONS")
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
    print("TRUE PORTFOLIO BACKTEST COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()