import pandas as pd


def generate_technical_signal(data: pd.DataFrame) -> dict:
    """
    Generate a technical BUY / HOLD / SELL signal.

    The signal is based on:
    - RSI
    - MACD
    - Moving averages
    - Bollinger Bands
    """

    latest = data.iloc[-1]

    score = 0

    reasons = []

    # --------------------------------
    # RSI
    # --------------------------------

    rsi = latest["RSI_14"]

    if rsi < 30:
        score += 2
        reasons.append("RSI indicates oversold conditions")

    elif rsi < 40:
        score += 1
        reasons.append("RSI indicates weak momentum")

    elif rsi > 70:
        score -= 2
        reasons.append("RSI indicates overbought conditions")

    elif rsi > 60:
        score -= 1
        reasons.append("RSI indicates strong but potentially stretched momentum")

    # --------------------------------
    # MACD
    # --------------------------------

    macd = latest["MACD"]

    macd_signal = latest["MACD_Signal"]

    if macd > macd_signal:
        score += 1
        reasons.append("MACD is above its signal line")

    else:
        score -= 1
        reasons.append("MACD is below its signal line")

    # --------------------------------
    # Moving Average
    # --------------------------------

    close = latest["Close"]

    sma_20 = latest["SMA_20"]

    sma_50 = latest["SMA_50"]

    if close > sma_20:
        score += 1
        reasons.append("Price is above the 20-day SMA")

    else:
        score -= 1
        reasons.append("Price is below the 20-day SMA")

    if sma_20 > sma_50:
        score += 1
        reasons.append("20-day SMA is above 50-day SMA")

    else:
        score -= 1
        reasons.append("20-day SMA is below 50-day SMA")

    # --------------------------------
    # Bollinger Bands
    # --------------------------------

    bb_upper = latest["BB_Upper"]

    bb_lower = latest["BB_Lower"]

    if close <= bb_lower:
        score += 1
        reasons.append("Price is near the lower Bollinger Band")

    elif close >= bb_upper:
        score -= 1
        reasons.append("Price is near the upper Bollinger Band")

    # --------------------------------
    # Final Signal
    # --------------------------------

    if score >= 3:
        signal = "BUY"

    elif score <= -3:
        signal = "SELL"

    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "score": score,
        "reasons": reasons
    }


if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]

    file_path = (
        project_root
        / "data"
        / "processed"
        / "RELIANCE_NS_features.csv"
    )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    result = generate_technical_signal(data)

    print("\nTechnical Signal")
    print("=" * 50)

    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']}")

    print("\nReasons:")

    for reason in result["reasons"]:
        print(f"- {reason}")