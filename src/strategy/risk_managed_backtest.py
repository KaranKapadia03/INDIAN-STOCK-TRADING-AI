from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "risk_managed_backtest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PARAMETERS
# =========================================================

TRAIN_SIZE = 0.60

TEST_WINDOW = 60

RETRAIN_EVERY = 60

INITIAL_CAPITAL = 100000

MAX_POSITION_SIZE = 0.20

MIN_POSITION_SIZE = 0.05

BUY_PROBABILITY = 0.60

HOLDING_PERIOD = 5

ATR_STOP_MULTIPLIER = 2.0

RISK_REWARD_RATIO = 2.0

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005


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
# TARGET
# =========================================================

def create_target(data):

    data = data.copy()

    data["Future_Return_5D"] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    data["Target"] = (
        data["Future_Return_5D"] > 1
    ).astype(int)

    data.dropna(
        subset=["Future_Return_5D"],
        inplace=True
    )

    return data


# =========================================================
# TECHNICAL SCORE
# =========================================================

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

    return np.clip(
        score / 6,
        -1,
        1
    )


# =========================================================
# RISK LEVELS
# =========================================================

def calculate_risk_levels(
    entry_price,
    atr_percent
):

    stop_distance = (
        atr_percent
        * ATR_STOP_MULTIPLIER
    )

    stop_loss = (
        entry_price
        * (1 - stop_distance)
    )

    target_distance = (
        stop_distance
        * RISK_REWARD_RATIO
    )

    take_profit = (
        entry_price
        * (1 + target_distance)
    )

    return (
        stop_loss,
        take_profit
    )


# =========================================================
# POSITION SIZE
# =========================================================

def calculate_position_size(
    probability,
    volatility
):

    if probability < BUY_PROBABILITY:
        return 0.0

    confidence = (
        probability
        - BUY_PROBABILITY
    ) / (
        1 - BUY_PROBABILITY
    )

    confidence = np.clip(
        confidence,
        0,
        1
    )

    volatility_factor = (
        1 / (1 + volatility)
    )

    position_size = (
        MIN_POSITION_SIZE
        +
        (
            MAX_POSITION_SIZE
            - MIN_POSITION_SIZE
        )
        * confidence
        * volatility_factor
    )

    return float(
        np.clip(
            position_size,
            MIN_POSITION_SIZE,
            MAX_POSITION_SIZE
        )
    )


# =========================================================
# MODEL
# =========================================================

def train_model(X, y):

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model


# =========================================================
# WALK-FORWARD PREDICTIONS
# =========================================================

def generate_predictions(data):

    predictions = []

    n = len(data)

    train_end = int(
        n * TRAIN_SIZE
    )

    while train_end < n:

        test_end = min(
            train_end + TEST_WINDOW,
            n
        )

        train_data = data.iloc[
            :train_end
        ]

        test_data = data.iloc[
            train_end:test_end
        ]

        if test_data.empty:
            break

        model = train_model(
            train_data[FEATURES],
            train_data["Target"]
        )

        probabilities = (
            model.predict_proba(
                test_data[FEATURES]
            )[:, 1]
        )

        for date, probability in zip(
            test_data.index,
            probabilities
        ):

            row = data.loc[date]

            technical = (
                calculate_technical_score(
                    row
                )
            )

            ml_score = (
                probability * 2 - 1
            )

            final_score = (
                ml_score * 0.625
                +
                technical * 0.375
            )

            if (
                probability >= BUY_PROBABILITY
                and final_score > 0
            ):
                signal = "BUY"
            else:
                signal = "HOLD"

            predictions.append(
                {
                    "date": date,
                    "ML_Probability":
                        probability,
                    "Technical_Score":
                        technical,
                    "Final_Score":
                        final_score,
                    "Signal":
                        signal,
                    "ATR_Percent":
                        row["ATR_Percent"],
                    "Volatility_20D":
                        row["Volatility_20D"],
                    "Future_Return_5D":
                        row["Future_Return_5D"],
                }
            )

        train_end += RETRAIN_EVERY

    return pd.DataFrame(
        predictions
    )


# =========================================================
# BACKTEST
# =========================================================

def backtest_stock(
    symbol,
    data,
    predictions
):

    capital = INITIAL_CAPITAL

    trades = []

    prediction_lookup = (
        predictions
        .set_index("date")
    )

    dates = list(
        prediction_lookup.index
    )

    i = 0

    while i < len(dates):

        signal_date = dates[i]

        signal_row = (
            prediction_lookup
            .loc[signal_date]
        )

        if signal_row["Signal"] != "BUY":

            i += 1
            continue

        # -------------------------------------------------
        # ENTRY
        # -------------------------------------------------

        signal_position = (
            data.index.get_loc(
                signal_date
            )
        )

        entry_position = (
            signal_position + 1
        )

        if entry_position >= len(data):
            break

        entry_date = data.index[
            entry_position
        ]

        entry_open = float(
            data.iloc[
                entry_position
            ]["Open"]
        )

        entry_price = (
            entry_open
            * (1 + SLIPPAGE)
        )

        # -------------------------------------------------
        # POSITION SIZE
        # -------------------------------------------------

        position_size = (
            calculate_position_size(
                signal_row[
                    "ML_Probability"
                ],
                signal_row[
                    "Volatility_20D"
                ]
            )
        )

        if position_size <= 0:

            i += 1
            continue

        allocation = (
            capital
            * position_size
        )

        # -------------------------------------------------
        # STOP LOSS / TAKE PROFIT
        # -------------------------------------------------

        stop_loss, take_profit = (
            calculate_risk_levels(
                entry_price,
                signal_row[
                    "ATR_Percent"
                ]
            )
        )

        # -------------------------------------------------
        # EXIT SEARCH
        # -------------------------------------------------

        max_exit_position = min(
            entry_position
            + HOLDING_PERIOD,
            len(data) - 1
        )

        exit_position = (
            max_exit_position
        )

        exit_reason = "TIME_EXIT"

        for position in range(
            entry_position,
            max_exit_position + 1
        ):

            day = data.iloc[
                position
            ]

            day_low = float(
                day["Low"]
            )

            day_high = float(
                day["High"]
            )

            # Conservative assumption:
            # if both are touched,
            # stop loss happens first.

            if day_low <= stop_loss:

                exit_position = position

                exit_price = stop_loss

                exit_reason = (
                    "STOP_LOSS"
                )

                break

            if day_high >= take_profit:

                exit_position = position

                exit_price = take_profit

                exit_reason = (
                    "TAKE_PROFIT"
                )

                break

        else:

            exit_price = float(
                data.iloc[
                    exit_position
                ]["Close"]
            )

        exit_date = data.index[
            exit_position
        ]

        exit_price *= (
            1 - SLIPPAGE
        )

        # -------------------------------------------------
        # RETURN
        # -------------------------------------------------

        gross_return = (
            exit_price
            / entry_price
            - 1
        )

        gross_profit = (
            allocation
            * gross_return
        )

        entry_cost = (
            allocation
            * TRANSACTION_COST
        )

        exit_cost = (
            allocation
            * TRANSACTION_COST
        )

        net_profit = (
            gross_profit
            - entry_cost
            - exit_cost
        )

        capital += net_profit

        trades.append(
            {
                "symbol": symbol,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size_pct":
                    position_size * 100,
                "allocation": allocation,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_reason": exit_reason,
                "gross_return_pct":
                    gross_return * 100,
                "net_profit": net_profit,
                "ML_Probability":
                    signal_row[
                        "ML_Probability"
                    ],
                "Technical_Score":
                    signal_row[
                        "Technical_Score"
                    ],
                "Final_Score":
                    signal_row[
                        "Final_Score"
                    ],
                "ATR_Percent":
                    signal_row[
                        "ATR_Percent"
                    ],
            }
        )

        # -------------------------------------------------
        # SKIP UNTIL TRADE CLOSES
        # -------------------------------------------------

        while (
            i < len(dates)
            and dates[i] <= exit_date
        ):
            i += 1

    return (
        capital,
        pd.DataFrame(trades)
    )


# =========================================================
# PROCESS STOCK
# =========================================================

def process_stock(filepath):

    symbol = (
        filepath.stem
        .replace(
            "_features",
            ""
        )
        .replace(
            "_",
            "."
        )
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

    data = create_target(
        data
    )

    predictions = (
        generate_predictions(
            data
        )
    )

    if predictions.empty:
        return None, None

    predictions.to_csv(
        OUTPUT_DIR
        / (
            symbol.replace(".", "_")
            + "_predictions.csv"
        ),
        index=False
    )

    final_capital, trades = (
        backtest_stock(
            symbol,
            data,
            predictions
        )
    )

    if trades.empty:

        metrics = {
            "symbol": symbol,
            "initial_capital":
                INITIAL_CAPITAL,
            "final_capital":
                INITIAL_CAPITAL,
            "return_pct": 0,
            "trades": 0,
            "win_rate": 0,
            "average_trade_pct": 0,
            "max_loss_pct": 0,
            "stop_loss_count": 0,
            "take_profit_count": 0,
        }

        return metrics, trades

    strategy_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1
    )

    win_rate = (
        trades["gross_return_pct"] > 0
    ).mean()

    average_trade = (
        trades["gross_return_pct"].mean()
    )

    max_loss = (
        trades["gross_return_pct"].min()
    )

    stop_count = (
        trades["exit_reason"]
        == "STOP_LOSS"
    ).sum()

    target_count = (
        trades["exit_reason"]
        == "TAKE_PROFIT"
    ).sum()

    metrics = {
        "symbol": symbol,
        "initial_capital":
            INITIAL_CAPITAL,
        "final_capital":
            final_capital,
        "return_pct":
            strategy_return * 100,
        "trades":
            len(trades),
        "win_rate":
            win_rate * 100,
        "average_trade_pct":
            average_trade,
        "max_loss_pct":
            max_loss,
        "stop_loss_count":
            stop_count,
        "take_profit_count":
            target_count,
    }

    return metrics, trades


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("RISK-MANAGED WALK-FORWARD BACKTEST")
    print("=" * 80)

    print(
        f"\nInitial capital: "
        f"₹{INITIAL_CAPITAL:,.0f}"
    )

    print(
        f"Maximum position size: "
        f"{MAX_POSITION_SIZE * 100:.0f}%"
    )

    print(
        f"Minimum position size: "
        f"{MIN_POSITION_SIZE * 100:.0f}%"
    )

    print(
        f"Buy probability: "
        f"{BUY_PROBABILITY * 100:.0f}%"
    )

    print(
        f"Holding period: "
        f"{HOLDING_PERIOD} days"
    )

    print(
        f"ATR stop multiplier: "
        f"{ATR_STOP_MULTIPLIER}"
    )

    print(
        f"Risk/reward ratio: "
        f"{RISK_REWARD_RATIO}:1"
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

            print(
                f"\nProcessing "
                f"{filepath.name}..."
            )

            metrics, trades = (
                process_stock(
                    filepath
                )
            )

            if metrics is None:
                continue

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
                f"{metrics['win_rate']:.2f}% | "
                f"Stops: "
                f"{metrics['stop_loss_count']} | "
                f"Targets: "
                f"{metrics['take_profit_count']}"
            )

        except Exception as e:

            print(
                f"ERROR - "
                f"{filepath.name}: "
                f"{e}"
            )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results = pd.DataFrame(
        all_metrics
    )

    results.to_csv(
        OUTPUT_DIR
        / "risk_managed_results.csv",
        index=False
    )

    if all_trades:

        trades_df = pd.concat(
            all_trades,
            ignore_index=True
        )

        trades_df.to_csv(
            OUTPUT_DIR
            / "risk_managed_trades.csv",
            index=False
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n")
    print("=" * 80)
    print("RISK-MANAGED RESULTS")
    print("=" * 80)

    if results.empty:

        print(
            "No results generated."
        )

        return

    print(
        f"\nAverage Return: "
        f"{results['return_pct'].mean():.2f}%"
    )

    print(
        f"Median Return: "
        f"{results['return_pct'].median():.2f}%"
    )

    print(
        f"Average Win Rate: "
        f"{results['win_rate'].mean():.2f}%"
    )

    print(
        f"Total Trades: "
        f"{results['trades'].sum()}"
    )

    print(
        f"Total Stop Losses: "
        f"{results['stop_loss_count'].sum()}"
    )

    print(
        f"Total Take Profits: "
        f"{results['take_profit_count'].sum()}"
    )

    print(
        f"Average Maximum Trade Loss: "
        f"{results['max_loss_pct'].mean():.2f}%"
    )

    print("\n")
    print("=" * 80)
    print("STOCK-BY-STOCK RESULTS")
    print("=" * 80)

    print(
        results[
            [
                "symbol",
                "return_pct",
                "trades",
                "win_rate",
                "average_trade_pct",
                "max_loss_pct",
                "stop_loss_count",
                "take_profit_count",
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
    print("RISK-MANAGED BACKTEST COMPLETE")
    print("=" * 80)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()