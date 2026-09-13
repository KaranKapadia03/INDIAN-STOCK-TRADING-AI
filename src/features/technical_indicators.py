import pandas as pd


def add_technical_indicators(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add technical and price-based features
    used by the trading AI.
    """

    df = data.copy()

    # ==================================================
    # MOVING AVERAGES
    # ==================================================

    df["SMA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # ==================================================
    # PRICE RETURNS
    # ==================================================

    df["Return_1D"] = (
        df["Close"].pct_change(1)
    )

    df["Return_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Return_10D"] = (
        df["Close"].pct_change(10)
    )

    df["Return_20D"] = (
        df["Close"].pct_change(20)
    )

    # ==================================================
    # VOLATILITY
    # ==================================================

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(window=20)
        .std()
    )

    # ==================================================
    # RSI
    # ==================================================

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = avg_gain / avg_loss

    df["RSI_14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # ==================================================
    # MACD
    # ==================================================

    ema_12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema_26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema_12 - ema_26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Histogram"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )

    # ==================================================
    # BOLLINGER BANDS
    # ==================================================

    rolling_mean = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(window=20)
        .std()
    )

    df["BB_Middle"] = rolling_mean

    df["BB_Upper"] = (
        rolling_mean
        + (2 * rolling_std)
    )

    df["BB_Lower"] = (
        rolling_mean
        - (2 * rolling_std)
    )

    # Position inside Bollinger Bands
    df["BB_Position"] = (
        (
            df["Close"]
            - df["BB_Lower"]
        )
        / (
            df["BB_Upper"]
            - df["BB_Lower"]
        )
    )

    # ==================================================
    # ATR
    # ==================================================

    high_low = (
        df["High"]
        - df["Low"]
    )

    high_close = (
        df["High"]
        - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"]
        - df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = (
        true_range
        .rolling(window=14)
        .mean()
    )

    # ATR relative to price
    df["ATR_Percent"] = (
        df["ATR_14"]
        / df["Close"]
    )

    # ==================================================
    # VOLUME FEATURES
    # ==================================================

    df["Volume_SMA_20"] = (
        df["Volume"]
        .rolling(window=20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        / df["Volume_SMA_20"]
    )

    # ==================================================
    # PRICE VS MOVING AVERAGES
    # ==================================================

    df["Price_vs_SMA20"] = (
        df["Close"]
        / df["SMA_20"]
        - 1
    )

    df["Price_vs_SMA50"] = (
        df["Close"]
        / df["SMA_50"]
        - 1
    )

    df["SMA20_vs_SMA50"] = (
        df["SMA_20"]
        / df["SMA_50"]
        - 1
    )

    return df


if __name__ == "__main__":

    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "RELIANCE_NS.csv"
    )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    data = add_technical_indicators(
        data
    )

    print(
        "\nTechnical Features"
    )

    print("=" * 60)

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print(
        "\nLatest features:"
    )

    print(
        data.tail(5).to_string()
    )