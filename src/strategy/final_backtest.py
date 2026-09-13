from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "trade_quality"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "final_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# STRATEGY PARAMETERS
# =========================================================

ML_WEIGHT = 0.50
TECHNICAL_WEIGHT = 0.50

BUY_PROBABILITY = 0.60

HOLDING_PERIOD = 5

INITIAL_CAPITAL = 100000

POSITION_SIZE = 0.20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005


# =========================================================
# TECHNICAL SCORE
# =========================================================

def technical_score(data):

    score = np.zeros(len(data))

    # RSI
    rsi = data["RSI_14"]

    score += np.where(
        rsi < 30,
        2,
        np.where(
            rsi < 40,
            1,
            np.where(
                rsi > 70,
                -2,
                np.where(
                    rsi > 60,
                    -1,
                    0
                )
            )
        )
    )

    # MACD
    score += np.where(
        data["MACD"]
        > data["MACD_Signal"],
        1,
        -1
    )

    # Price vs SMA20
    score += np.where(
        data["Close"]
        > data["SMA_20"],
        1,
        -1
    )

    # SMA20 vs SMA50
    score += np.where(
        data["SMA_20"]
        > data["SMA_50"],
        1,
        -1
    )

    # Bollinger
    score += np.where(
        data["Close"]
        <= data["BB_Lower"],
        1,
        np.where(
            data["Close"]
            >= data["BB_Upper"],
            -1,
            0
        )
    )

    return np.clip(
        score / 6,
        -1,
        1
    )


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(symbol):

    model_path = (
        MODEL_DIR
        / symbol.replace(".", "_")
        / "trade_quality_model.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def prepare_data(filepath):

    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(
        inplace=True
    )

    data.dropna(
        subset=FEATURES,
        inplace=True
    )

    # Use exactly the same chronological
    # 80/20 split as model training.

    split_index = int(
        len(data) * 0.80
    )

    test_data = data.iloc[
        split_index:
    ].copy()

    model = load_model(
        symbol
    )

    # ML probability
    test_data[
        "ML_Probability"
    ] = model.predict_proba(
        test_data[FEATURES]
    )[:, 1]

    # Technical score
    test_data[
        "Technical_Score"
    ] = technical_score(
        test_data
    )

    # Convert ML probability
    # into directional score.

    test_data[
        "ML_Score"
    ] = (
        test_data["ML_Probability"]
        * 2
        - 1
    )

    # Final score
    test_data[
        "Final_Score"
    ] = (
        test_data["ML_Score"]
        * ML_WEIGHT
        +
        test_data["Technical_Score"]
        * TECHNICAL_WEIGHT
    )

    # BUY only when both:
    #
    # ML probability >= 60%
    # Final score is positive

    test_data[
        "Signal"
    ] = np.where(
        (
            test_data["ML_Probability"]
            >= BUY_PROBABILITY
        )
        &
        (
            test_data["Final_Score"]
            > 0
        ),
        "BUY",
        "HOLD"
    )

    return symbol, test_data


# =========================================================
# BACKTEST
# =========================================================

def backtest_stock(
    symbol,
    data
):

    capital = INITIAL_CAPITAL

    equity_curve = []

    trades = []

    i = 0

    while i < len(data):

        row = data.iloc[i]

        date = data.index[i]

        # -------------------------------------------------
        # NO TRADE
        # -------------------------------------------------

        if row["Signal"] != "BUY":

            equity_curve.append(
                {
                    "date": date,
                    "capital": capital,
                }
            )

            i += 1

            continue

        # -------------------------------------------------
        # ENTRY
        # -------------------------------------------------

        entry_price = float(
            row["Close"]
        )

        # Slippage on entry
        entry_price *= (
            1 + SLIPPAGE
        )

        allocation = (
            capital
            * POSITION_SIZE
        )

        # Transaction cost
        entry_cost = (
            allocation
            * TRANSACTION_COST
        )

        capital_after_entry = (
            capital
            - entry_cost
        )

        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        exit_index = (
            i + HOLDING_PERIOD
        )

        if exit_index >= len(data):

            break

        exit_row = data.iloc[
            exit_index
        ]

        exit_date = data.index[
            exit_index
        ]

        exit_price = float(
            exit_row["Close"]
        )

        # Slippage on exit
        exit_price *= (
            1 - SLIPPAGE
        )

        stock_return = (
            exit_price
            / entry_price
            - 1
        )

        gross_profit = (
            allocation
            * stock_return
        )

        exit_cost = (
            allocation
            * abs(stock_return)
            * TRANSACTION_COST
        )

        net_profit = (
            gross_profit
            - entry_cost
            - exit_cost
        )

        capital = (
            capital
            + net_profit
        )

        trades.append(
            {
                "symbol": symbol,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": stock_return,
                "net_profit": net_profit,
                "ml_probability":
                    row["ML_Probability"],
                "technical_score":
                    row["Technical_Score"],
                "final_score":
                    row["Final_Score"],
            }
        )

        # -------------------------------------------------
        # EQUITY
        # -------------------------------------------------

        equity_curve.append(
            {
                "date": exit_date,
                "capital": capital,
            }
        )

        # Skip forward until trade exits.
        i = exit_index + 1

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    trades_df = pd.DataFrame(
        trades
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    if trades_df.empty:

        return {
            "symbol": symbol,
            "initial_capital":
                INITIAL_CAPITAL,
            "final_capital":
                INITIAL_CAPITAL,
            "return_pct": 0,
            "trades": 0,
            "win_rate": 0,
            "avg_trade": 0,
            "max_drawdown": 0,
            "sharpe": 0,
            "profit_factor": 0,
        }, trades_df, equity_df

    final_capital = capital

    strategy_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    )

    win_rate = (
        trades_df["return"] > 0
    ).mean()

    average_trade = (
        trades_df["return"].mean()
    )

    # -----------------------------------------------------
    # DRAWDOWN
    # -----------------------------------------------------

    if not equity_df.empty:

        equity_df[
            "peak"
        ] = equity_df[
            "capital"
        ].cummax()

        equity_df[
            "drawdown"
        ] = (
            equity_df["capital"]
            / equity_df["peak"]
            - 1
        )

        max_drawdown = (
            equity_df["drawdown"].min()
        )

    else:

        max_drawdown = 0

    # -----------------------------------------------------
    # SHARPE
    # -----------------------------------------------------

    trade_returns = (
        trades_df["return"]
    )

    if (
        len(trade_returns) > 1
        and trade_returns.std() != 0
    ):

        sharpe = (
            trade_returns.mean()
            / trade_returns.std()
            * np.sqrt(252 / HOLDING_PERIOD)
        )

    else:

        sharpe = 0

    # -----------------------------------------------------
    # PROFIT FACTOR
    # -----------------------------------------------------

    profits = (
        trades_df.loc[
            trades_df["net_profit"] > 0,
            "net_profit"
        ].sum()
    )

    losses = abs(
        trades_df.loc[
            trades_df["net_profit"] < 0,
            "net_profit"
        ].sum()
    )

    if losses > 0:

        profit_factor = (
            profits / losses
        )

    else:

        profit_factor = np.inf

    metrics = {
        "symbol": symbol,
        "initial_capital":
            INITIAL_CAPITAL,
        "final_capital":
            final_capital,
        "return_pct":
            strategy_return * 100,
        "trades":
            len(trades_df),
        "win_rate":
            win_rate * 100,
        "avg_trade":
            average_trade * 100,
        "max_drawdown":
            max_drawdown * 100,
        "sharpe":
            sharpe,
        "profit_factor":
            profit_factor,
    }

    return (
        metrics,
        trades_df,
        equity_df
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("FINAL STRATEGY BACKTEST")
    print("=" * 80)

    print(
        f"\nInitial capital: "
        f"₹{INITIAL_CAPITAL:,.0f}"
    )

    print(
        f"Position size: "
        f"{POSITION_SIZE * 100:.0f}%"
    )

    print(
        f"Holding period: "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.2f}%"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE * 100:.2f}%"
    )

    files = sorted(
        FEATURES_DIR.glob(
            "*_features.csv"
        )
    )

    all_metrics = []

    all_trades = []

    for filepath in files:

        try:

            symbol, data = prepare_data(
                filepath
            )

            print(
                f"\nBacktesting {symbol}..."
            )

            metrics, trades, equity = (
                backtest_stock(
                    symbol,
                    data
                )
            )

            all_metrics.append(
                metrics
            )

            if not trades.empty:

                all_trades.append(
                    trades
                )

            print(
                f"Return: "
                f"{metrics['return_pct']:.2f}% | "
                f"Trades: "
                f"{metrics['trades']} | "
                f"Win Rate: "
                f"{metrics['win_rate']:.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR - {filepath.name}: {e}"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    if metrics_df.empty:

        print(
            "\nNo backtest results."
        )

        return

    metrics_file = (
        OUTPUT_DIR
        / "final_strategy_performance.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    if all_trades:

        trades_df = pd.concat(
            all_trades,
            ignore_index=True
        )

        trades_df.to_csv(
            OUTPUT_DIR
            / "final_strategy_trades.csv",
            index=False
        )

    # =====================================================
    # OVERALL
    # =====================================================

    print("\n")
    print("=" * 80)
    print("FINAL STRATEGY RESULTS")
    print("=" * 80)

    print(
        f"\nAverage Return: "
        f"{metrics_df['return_pct'].mean():.2f}%"
    )

    print(
        f"Median Return: "
        f"{metrics_df['return_pct'].median():.2f}%"
    )

    print(
        f"Average Win Rate: "
        f"{metrics_df['win_rate'].mean():.2f}%"
    )

    print(
        f"Average Trades: "
        f"{metrics_df['trades'].mean():.1f}"
    )

    print(
        f"Average Drawdown: "
        f"{metrics_df['max_drawdown'].mean():.2f}%"
    )

    print(
        f"Average Sharpe: "
        f"{metrics_df['sharpe'].mean():.2f}"
    )

    print(
        f"Average Profit Factor: "
        f"{metrics_df['profit_factor'].replace(
            [np.inf, -np.inf],
            np.nan
        ).mean():.2f}"
    )

    # =====================================================
    # STOCK RESULTS
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STOCK-BY-STOCK RESULTS")
    print("=" * 80)

    print(
        metrics_df[
            [
                "symbol",
                "return_pct",
                "trades",
                "win_rate",
                "avg_trade",
                "max_drawdown",
                "sharpe",
                "profit_factor",
            ]
        ]
        .sort_values(
            "return_pct",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()