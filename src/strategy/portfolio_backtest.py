from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

INITIAL_CAPITAL = 100000
HOLDING_PERIOD = 5

# Number of stocks held at the same time
TOP_N_STOCKS = 5

# Equal capital allocation
POSITION_SIZE = 1 / TOP_N_STOCKS


FEATURE_COLUMNS = [
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
]


def load_stock_data(symbol):
    """
    Load processed features for one stock.
    """

    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    filepath = PROCESSED_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Feature file not found: {filepath}"
        )

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(inplace=True)

    return data


def load_model(symbol):
    """
    Load the trained ML model and encoder.
    """

    model_dir = (
        MODEL_DIR
        / symbol.replace(".", "_")
    )

    model = joblib.load(
        model_dir / "model.pkl"
    )

    encoder = joblib.load(
        model_dir / "label_encoder.pkl"
    )

    return model, encoder


def create_target(data):
    """
    Create the 5-day future return.

    This is only used to evaluate
    the prediction after the fact.
    """

    df = data.copy()

    df["Future_Return_5D"] = (
        df["Close"].shift(-HOLDING_PERIOD)
        / df["Close"]
        - 1
    ) * 100

    df.dropna(
        subset=["Future_Return_5D"],
        inplace=True
    )

    return df


def generate_predictions(symbol):
    """
    Generate ML predictions for the
    unseen 20% test period.
    """

    data = load_stock_data(symbol)

    data = create_target(data)

    split_index = int(
        len(data) * 0.80
    )

    test_data = data.iloc[
        split_index:
    ].copy()

    model, encoder = load_model(symbol)

    X_test = test_data[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    predictions = encoder.inverse_transform(
        predictions
    )

    test_data["Prediction"] = predictions

    return test_data


def calculate_sharpe(returns):
    """
    Calculate annualized Sharpe ratio.
    """

    if len(returns) < 2:
        return 0

    if returns.std() == 0:
        return 0

    return (
        returns.mean()
        / returns.std()
    ) * np.sqrt(252)


def calculate_max_drawdown(equity):
    """
    Calculate maximum drawdown.
    """

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    return drawdown.min() * 100


def build_prediction_matrix(stocks):
    """
    Generate predictions for all stocks
    and align them by trading date.
    """

    prediction_data = {}

    for symbol in stocks:

        try:

            print(
                f"Generating predictions: "
                f"{symbol}"
            )

            prediction_data[symbol] = (
                generate_predictions(symbol)
            )

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

    return prediction_data


def run_portfolio_backtest(
    prediction_data
):
    """
    Portfolio backtest.

    Every trading day:

    1. Look at all stock predictions.
    2. Select stocks predicted UP.
    3. Rank them by model confidence.
    4. Select the top N stocks.
    5. Allocate capital equally.
    6. Hold each position for 5 trading days.
    """

    # --------------------------------------------------
    # Find dates common to all stocks
    # --------------------------------------------------

    common_dates = None

    for data in prediction_data.values():

        dates = set(data.index)

        if common_dates is None:
            common_dates = dates
        else:
            common_dates &= dates

    common_dates = sorted(common_dates)

    if not common_dates:
        raise ValueError(
            "No common trading dates found."
        )

    # --------------------------------------------------
    # Portfolio state
    # --------------------------------------------------

    capital = INITIAL_CAPITAL

    portfolio_history = []

    trades = []

    i = 0

    while i < len(common_dates):

        current_date = common_dates[i]

        candidates = []

        # --------------------------------------------------
        # Examine every stock
        # --------------------------------------------------

        for symbol, data in prediction_data.items():

            if current_date not in data.index:
                continue

            row = data.loc[current_date]

            if row["Prediction"] != "UP":
                continue

            # Model probability
            model, encoder = load_model(
                symbol
            )

            feature_row = row[
                FEATURE_COLUMNS
            ].to_frame().T

            probabilities = (
                model.predict_proba(
                    feature_row
                )[0]
            )

            predicted_class = model.predict(
                feature_row
            )[0]

            predicted_class_name = (
                encoder.inverse_transform(
                    [predicted_class]
                )[0]
            )

            class_probabilities = {
                class_name: probability
                for class_name, probability
                in zip(
                    encoder.classes_,
                    probabilities
                )
            }

            confidence = class_probabilities.get(
                predicted_class_name,
                0
            )

            candidates.append({
                "symbol": symbol,
                "confidence": confidence,
                "entry_price": row["Close"],
                "data": data,
            })

        # --------------------------------------------------
        # No BUY opportunities
        # --------------------------------------------------

        if not candidates:

            portfolio_history.append({
                "Date": current_date,
                "Capital": capital,
                "Daily Return": 0,
            })

            i += 1

            continue

        # --------------------------------------------------
        # Rank by model confidence
        # --------------------------------------------------

        candidates.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        selected = candidates[
            :TOP_N_STOCKS
        ]

        # --------------------------------------------------
        # Execute positions
        # --------------------------------------------------

        trade_capital = capital

        portfolio_return = 0

        max_exit_index = i

        for candidate in selected:

            symbol = candidate["symbol"]

            data = candidate["data"]

            # Find current position
            current_position = data.index.get_loc(
                current_date
            )

            exit_position = (
                current_position
                + HOLDING_PERIOD
            )

            if exit_position >= len(data):
                continue

            exit_date = data.index[
                exit_position
            ]

            entry_price = (
                candidate["entry_price"]
            )

            exit_price = data.iloc[
                exit_position
            ]["Close"]

            trade_return = (
                exit_price
                / entry_price
                - 1
            )

            weighted_return = (
                trade_return
                * POSITION_SIZE
            )

            portfolio_return += (
                weighted_return
            )

            trades.append({
                "Entry Date": current_date,
                "Exit Date": exit_date,
                "Symbol": symbol,
                "Confidence": candidate[
                    "confidence"
                ],
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": trade_return * 100,
                "Portfolio Weight %":
                    POSITION_SIZE * 100,
            })

            max_exit_index = max(
                max_exit_index,
                common_dates.index(exit_date)
                if exit_date in common_dates
                else i
            )

        # --------------------------------------------------
        # Update capital
        # --------------------------------------------------

        capital *= (
            1 + portfolio_return
        )

        portfolio_history.append({
            "Date": current_date,
            "Capital": capital,
            "Daily Return": portfolio_return,
        })

        # --------------------------------------------------
        # Move forward.
        #
        # We do not create another portfolio
        # while these positions are open.
        # --------------------------------------------------

        i += HOLDING_PERIOD

    # --------------------------------------------------
    # Create equity curve
    # --------------------------------------------------

    history = pd.DataFrame(
        portfolio_history
    )

    history["Date"] = pd.to_datetime(
        history["Date"]
    )

    history.set_index(
        "Date",
        inplace=True
    )

    equity = history["Capital"]

    portfolio_returns = (
        history["Daily Return"]
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    strategy_return = (
        capital / INITIAL_CAPITAL
        - 1
    ) * 100

    max_drawdown = calculate_max_drawdown(
        equity
    )

    sharpe = calculate_sharpe(
        portfolio_returns
    )

    total_trades = len(trades)

    if total_trades > 0:

        winning_trades = sum(
            trade["Return %"] > 0
            for trade in trades
        )

        win_rate = (
            winning_trades
            / total_trades
        ) * 100

        average_trade_return = np.mean(
            [
                trade["Return %"]
                for trade in trades
            ]
        )

    else:

        win_rate = 0
        average_trade_return = 0

    return {
        "final_capital": capital,
        "strategy_return": strategy_return,
        "trades": total_trades,
        "win_rate": win_rate,
        "average_trade_return":
            average_trade_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "history": history,
        "trades_log": trades,
    }


def main():

    print("\n" + "=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "PORTFOLIO-LEVEL ML BACKTEST"
    )
    print("=" * 80)

    # --------------------------------------------------
    # Find all trained stock models
    # --------------------------------------------------

    model_directories = [
        path
        for path in MODEL_DIR.iterdir()
        if path.is_dir()
        and (path / "model.pkl").exists()
    ]

    stocks = sorted([
        path.name.replace("_", ".")
        for path in model_directories
    ])

    print(
        f"\nStocks available: {len(stocks)}"
    )

    print(
        f"Initial Capital: "
        f"₹{INITIAL_CAPITAL:,.0f}"
    )

    print(
        f"Portfolio Size: "
        f"Top {TOP_N_STOCKS} stocks"
    )

    print(
        f"Holding Period: "
        f"{HOLDING_PERIOD} trading days"
    )

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    prediction_data = (
        build_prediction_matrix(
            stocks
        )
    )

    print(
        f"\nSuccessfully loaded "
        f"{len(prediction_data)} stock models."
    )

    # --------------------------------------------------
    # Run portfolio backtest
    # --------------------------------------------------

    results = run_portfolio_backtest(
        prediction_data
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "PORTFOLIO BACKTEST RESULTS"
    )
    print("=" * 80)

    print(
        f"Initial Capital:      "
        f"₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Capital:        "
        f"₹{results['final_capital']:,.2f}"
    )

    print(
        f"Strategy Return:      "
        f"{results['strategy_return']:.2f}%"
    )

    print(
        f"Total Trades:         "
        f"{results['trades']}"
    )

    print(
        f"Win Rate:             "
        f"{results['win_rate']:.2f}%"
    )

    print(
        f"Average Trade:        "
        f"{results['average_trade_return']:.2f}%"
    )

    print(
        f"Maximum Drawdown:     "
        f"{results['max_drawdown']:.2f}%"
    )

    print(
        f"Sharpe Ratio:         "
        f"{results['sharpe']:.2f}"
    )

    # --------------------------------------------------
    # Save portfolio equity curve
    # --------------------------------------------------

    equity_file = (
        MODEL_DIR
        / "portfolio_equity_curve.csv"
    )

    results["history"].to_csv(
        equity_file
    )

    # --------------------------------------------------
    # Save trade log
    # --------------------------------------------------

    trade_file = (
        MODEL_DIR
        / "portfolio_trade_log.csv"
    )

    pd.DataFrame(
        results["trades_log"]
    ).to_csv(
        trade_file,
        index=False
    )

    print("\nSaved:")

    print(
        equity_file
    )

    print(
        trade_file
    )


if __name__ == "__main__":
    main()