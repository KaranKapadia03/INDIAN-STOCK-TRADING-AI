
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY / HOLD / SELL signals using technical indicators.
    """

    df = data.copy()

    df["Signal"] = "HOLD"

    buy_condition = (
        (df["Close"] > df["SMA_20"])
        & (df["SMA_20"] > df["SMA_50"])
        & (df["MACD"] > df["MACD_Signal"])
    )

    sell_condition = (
        (df["Close"] < df["SMA_20"])
        & (df["SMA_20"] < df["SMA_50"])
        & (df["MACD"] < df["MACD_Signal"])
    )

    df.loc[buy_condition, "Signal"] = "BUY"

    df.loc[sell_condition, "Signal"] = "SELL"

    return df


def create_positions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signals into investment positions.

    BUY  -> 1 (invested)
    SELL -> 0 (cash)
    HOLD -> keep previous position
    """

    df = data.copy()

    position = 0

    positions = []

    for signal in df["Signal"]:

        if signal == "BUY":
            position = 1

        elif signal == "SELL":
            position = 0

        positions.append(position)

    df["Position"] = positions

    return df


def calculate_strategy_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily strategy returns.

    The previous day's position is used so that today's
    signal does not magically earn today's return.
    """

    df = data.copy()

    df["Market_Return"] = (
        df["Close"].pct_change()
    )

    df["Strategy_Return"] = (
        df["Position"].shift(1)
        * df["Market_Return"]
    )

    df["Strategy_Return"] = (
        df["Strategy_Return"].fillna(0)
    )

    return df


def calculate_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate backtest performance metrics.
    """

    strategy_returns = data[
        "Strategy_Return"
    ]

    # --------------------------------
    # Strategy return
    # --------------------------------

    strategy_return = (
        (1 + strategy_returns).prod()
        - 1
    )

    # --------------------------------
    # Buy and hold
    # --------------------------------

    buy_hold_return = (
        data["Close"].iloc[-1]
        / data["Close"].iloc[0]
        - 1
    )

    # --------------------------------
    # Number of trades
    # --------------------------------

    position_changes = (
        data["Position"].diff().abs()
    )

    trades = int(
        position_changes.sum()
    )

    # --------------------------------
    # Winning days
    # --------------------------------

    active_returns = strategy_returns[
        data["Position"].shift(1) == 1
    ]

    if len(active_returns) > 0:

        win_rate = (
            (active_returns > 0).mean()
        )

    else:

        win_rate = 0

    # --------------------------------
    # Maximum drawdown
    # --------------------------------

    equity_curve = (
        1 + strategy_returns
    ).cumprod()

    running_max = (
        equity_curve.cummax()
    )

    drawdown = (
        equity_curve
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    # --------------------------------
    # Sharpe ratio
    # --------------------------------

    if strategy_returns.std() != 0:

        sharpe_ratio = (
            strategy_returns.mean()
            / strategy_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe_ratio = 0

    # --------------------------------
    # Time invested
    # --------------------------------

    time_in_market = (
        data["Position"].mean()
    )

    return {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "trades": trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "time_in_market": time_in_market,
    }


def main():

    symbol = "RELIANCE.NS"

    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    file_path = (
        PROCESSED_DATA_DIR
        / filename
    )

    print("\nLoading data...")

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(inplace=True)

    print(
        f"Rows: {len(data)}"
    )

    # Generate technical signals
    data = generate_signals(data)

    # Convert signals into positions
    data = create_positions(data)

    # Calculate returns
    data = calculate_strategy_returns(data)

    # Calculate metrics
    metrics = calculate_metrics(data)

    print(
        "\n" + "=" * 60
    )

    print(
        "RELIANCE BACKTEST"
    )

    print(
        "=" * 60
    )

    print(
        f"\nStrategy Return: "
        f"{metrics['strategy_return']:.2%}"
    )

    print(
        f"Buy & Hold Return: "
        f"{metrics['buy_hold_return']:.2%}"
    )

    print(
        f"Number of Trades: "
        f"{metrics['trades']}"
    )

    print(
        f"Win Rate: "
        f"{metrics['win_rate']:.2%}"
    )

    print(
        f"Maximum Drawdown: "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe Ratio: "
        f"{metrics['sharpe_ratio']:.2f}"
    )

    print(
        f"Time in Market: "
        f"{metrics['time_in_market']:.2%}"
    )

    print(
        "\nRecent signals:"
    )

    print(
        data[
            [
                "Close",
                "Signal",
                "Position",
                "Strategy_Return",
            ]
        ].tail(10)
    )


if __name__ == "__main__":
    main()

