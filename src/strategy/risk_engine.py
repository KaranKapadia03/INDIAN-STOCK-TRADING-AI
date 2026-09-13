import numpy as np
import pandas as pd


# ============================================================
# RISK MANAGEMENT SETTINGS
# ============================================================

MAX_POSITION_SIZE = 0.20      # Maximum 20% of portfolio in one stock

MIN_CONFIDENCE = 0.50         # Minimum ML confidence required
MAX_CONFIDENCE = 0.90         # Confidence considered very strong


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(data, window=20):
    """
    Calculate annualized volatility using recent daily returns.

    Example:
    0.25 = approximately 25% annualized volatility
    """

    returns = data["Close"].pct_change()

    volatility = (
        returns
        .rolling(window)
        .std()
        * np.sqrt(252)
    )

    return volatility.iloc[-1]


# ============================================================
# ATR
# ============================================================

def calculate_atr_percent(data):
    """
    Calculate Average True Range as a percentage of price.

    This tells us how much the stock typically moves
    relative to its current price.
    """

    high_low = (
        data["High"] - data["Low"]
    )

    high_close = (
        data["High"]
        - data["Close"].shift(1)
    ).abs()

    low_close = (
        data["Low"]
        - data["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(14)
        .mean()
    )

    atr_percent = (
        atr / data["Close"]
    )

    return atr_percent.iloc[-1]


# ============================================================
# POSITION SIZING
# ============================================================

def calculate_position_size(
    ml_confidence,
    signal_strength,
    volatility,
    max_position=MAX_POSITION_SIZE
):
    """
    Calculate how much of the portfolio should be allocated
    to a stock.

    Position size depends on:

    1. ML confidence
    2. Final signal strength
    3. Stock volatility

    Higher confidence  -> larger position
    Stronger signal    -> larger position
    Higher volatility  -> smaller position
    """

    # --------------------------------------------------------
    # STEP 1: Check minimum ML confidence
    # --------------------------------------------------------

    if ml_confidence < MIN_CONFIDENCE:
        return 0.0

    # --------------------------------------------------------
    # STEP 2: Convert ML confidence into 0-1 score
    # --------------------------------------------------------

    confidence_score = (
        ml_confidence - MIN_CONFIDENCE
    ) / (
        MAX_CONFIDENCE - MIN_CONFIDENCE
    )

    confidence_score = np.clip(
        confidence_score,
        0,
        1
    )

    # --------------------------------------------------------
    # STEP 3: Convert signal strength into 0-1 score
    # --------------------------------------------------------

    signal_score = np.clip(
        abs(signal_strength),
        0,
        1
    )

    # --------------------------------------------------------
    # STEP 4: Reduce allocation for volatile stocks
    # --------------------------------------------------------

    volatility_factor = (
        1 / (1 + volatility)
    )

    # --------------------------------------------------------
    # STEP 5: Calculate overall conviction
    #
    # ML confidence = 60%
    # Signal strength = 40%
    # --------------------------------------------------------

    conviction = (
        confidence_score * 0.60
        + signal_score * 0.40
    )

    # --------------------------------------------------------
    # STEP 6: Calculate position size
    # --------------------------------------------------------

    position_size = (
        max_position
        * conviction
        * volatility_factor
        * 1.5
    )

    # --------------------------------------------------------
    # STEP 7: Never exceed maximum position
    # --------------------------------------------------------

    position_size = min(
        position_size,
        max_position
    )

    return float(position_size)


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    entry_price,
    atr_percent,
    multiplier=2
):
    """
    Calculate ATR-based stop loss.

    Default:
    Stop loss = 2 × ATR below entry price
    """

    stop_distance = (
        atr_percent * multiplier
    )

    stop_loss = (
        entry_price
        * (1 - stop_distance)
    )

    return stop_loss


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_take_profit(
    entry_price,
    atr_percent,
    risk_reward_ratio=2
):
    """
    Calculate ATR-based take-profit level.

    Default risk/reward ratio = 1:2.
    """

    stop_distance = (
        atr_percent * 2
    )

    target_distance = (
        stop_distance
        * risk_reward_ratio
    )

    take_profit = (
        entry_price
        * (1 + target_distance)
    )

    return take_profit


# ============================================================
# COMPLETE RISK ASSESSMENT
# ============================================================

def assess_risk(
    data,
    ml_confidence,
    signal_strength,
    entry_price=None
):
    """
    Generate complete risk information for a stock.
    """

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    if entry_price is None:
        entry_price = data["Close"].iloc[-1]

    # --------------------------------------------------------
    # Calculate risk metrics
    # --------------------------------------------------------

    volatility = calculate_volatility(
        data
    )

    atr_percent = calculate_atr_percent(
        data
    )

    # --------------------------------------------------------
    # Calculate position size
    # --------------------------------------------------------

    position_size = calculate_position_size(
        ml_confidence=ml_confidence,
        signal_strength=signal_strength,
        volatility=volatility
    )

    # --------------------------------------------------------
    # Calculate stop loss
    # --------------------------------------------------------

    stop_loss = calculate_stop_loss(
        entry_price=entry_price,
        atr_percent=atr_percent
    )

    # --------------------------------------------------------
    # Calculate take profit
    # --------------------------------------------------------

    take_profit = calculate_take_profit(
        entry_price=entry_price,
        atr_percent=atr_percent
    )

    # --------------------------------------------------------
    # Return all risk information
    # --------------------------------------------------------

    return {
        "ml_confidence": ml_confidence,
        "signal_strength": signal_strength,
        "volatility": volatility,
        "atr_percent": atr_percent,
        "position_size": position_size,
        "position_size_percent": position_size * 100,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 60)
    print("INDIAN STOCK TRADING AI")
    print("RISK ENGINE")
    print("=" * 60)

    print(
        f"Maximum position size: "
        f"{MAX_POSITION_SIZE * 100:.0f}%"
    )

    print(
        f"Minimum ML confidence: "
        f"{MIN_CONFIDENCE * 100:.0f}%"
    )

    print("Risk engine loaded successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()