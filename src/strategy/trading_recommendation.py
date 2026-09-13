import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.strategy.technical_signal import (
    generate_technical_signal
)

from src.strategy.risk_engine import (
    assess_risk
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)

FEATURES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

NEWS_FILE = os.path.join(
    FEATURES_DIR,
    "news",
    "live_news_signals.csv"
)

TRADE_PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "current_trade_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "trading_recommendations.csv"
)


# ============================================================
# STRATEGY WEIGHTS
# ============================================================

ML_WEIGHT = 0.45
TECHNICAL_WEIGHT = 0.30
NEWS_WEIGHT = 0.15
RETURN_WEIGHT = 0.10


# ============================================================
# DECISION THRESHOLDS
# ============================================================

BUY_THRESHOLD = 0.20
SELL_THRESHOLD = -0.20

MIN_BUY_PROBABILITY = 0.60
MIN_SELL_PROBABILITY = 0.60

MIN_EXPECTED_RETURN = 2.0
MAX_EXPECTED_RETURN_FOR_SELL = -2.0


# ============================================================
# LOAD TRADE PREDICTIONS
# ============================================================

def load_trade_predictions():

    if not os.path.exists(
        TRADE_PREDICTIONS_FILE
    ):

        raise FileNotFoundError(
            f"Trade prediction file not found:\n"
            f"{TRADE_PREDICTIONS_FILE}\n\n"
            f"Run:\n"
            f"python src\\models\\predict_trade_models.py"
        )

    return pd.read_csv(
        TRADE_PREDICTIONS_FILE
    )


# ============================================================
# LOAD LIVE NEWS
# ============================================================

def load_live_news():

    if not os.path.exists(
        NEWS_FILE
    ):

        print(
            "\nWARNING: Live news file not found."
        )

        return pd.DataFrame()

    news = pd.read_csv(
        NEWS_FILE
    )

    return news


# ============================================================
# GET STOCK FEATURES
# ============================================================

def load_stock_features(symbol):

    file_path = os.path.join(
        FEATURES_DIR,
        f"{symbol.replace('.', '_')}_features_news.csv"
    )

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Feature file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    return df


# ============================================================
# NORMALIZE TECHNICAL SCORE
# ============================================================

def normalize_technical_score(
    technical_score
):

    # Technical signal normally ranges
    # approximately from -6 to +6.

    normalized = (
        technical_score / 6.0
    )

    return float(
        np.clip(
            normalized,
            -1,
            1
        )
    )


# ============================================================
# NORMALIZE NEWS SCORE
# ============================================================

def get_news_score(
    symbol,
    news
):

    if news.empty:

        return 0.0, 0, "NO NEWS"

    row = news[
        news["symbol"] == symbol
    ]

    if row.empty:

        return 0.0, 0, "NO NEWS"

    row = row.iloc[0]

    score = float(
        row.get(
            "News_Signal",
            0
        )
    )

    article_count = int(
        row.get(
            "News_Count",
            0
        )
    )

    label = str(
        row.get(
            "News_Label",
            "NEUTRAL"
        )
    )

    return (
        score,
        article_count,
        label
    )


# ============================================================
# EXPECTED RETURN SCORE
# ============================================================

def calculate_return_score(
    expected_return
):

    # Convert expected 20D return into a bounded score.
    #
    # +10% expected return -> +1
    # -10% expected return -> -1

    score = (
        expected_return / 10.0
    )

    return float(
        np.clip(
            score,
            -1,
            1
        )
    )


# ============================================================
# ML SCORE
# ============================================================

def calculate_ml_score(
    buy_probability,
    sell_probability
):

    # Strong BUY probability creates positive score.
    # Strong SELL probability creates negative score.

    buy_score = (
        buy_probability - 0.50
    ) * 2

    sell_score = (
        sell_probability - 0.50
    ) * 2

    score = buy_score - sell_score

    return float(
        np.clip(
            score,
            -1,
            1
        )
    )


# ============================================================
# QUALITY ADJUSTMENT
# ============================================================

def adjust_ml_score_for_quality(
    ml_score,
    ml_weight
):

    return float(
        ml_score * ml_weight
    )


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================

def generate_recommendation(
    prediction,
    technical_signal,
    news_score,
    news_articles,
    news_label,
    risk
):

    symbol = prediction[
        "symbol"
    ]

    buy_probability = float(
        prediction[
            "effective_buy_probability"
        ]
    )

    sell_probability = float(
        prediction[
            "effective_sell_probability"
        ]
    )

    expected_return = float(
        prediction[
            "expected_return_20d"
        ]
    )

    ml_weight = float(
        prediction[
            "ml_weight"
        ]
    )

    overall_quality = prediction[
        "overall_quality"
    ]

    # --------------------------------------------------------
    # Individual scores
    # --------------------------------------------------------

    raw_ml_score = calculate_ml_score(
        buy_probability,
        sell_probability
    )

    ml_score = adjust_ml_score_for_quality(
        raw_ml_score,
        ml_weight
    )

    technical_score = normalize_technical_score(
        technical_signal["score"]
    )

    return_score = calculate_return_score(
        expected_return
    )

    news_score = float(
        np.clip(
            news_score,
            -1,
            1
        )
    )

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------

    final_score = (

        ml_score * ML_WEIGHT

        + technical_score * TECHNICAL_WEIGHT

        + news_score * NEWS_WEIGHT

        + return_score * RETURN_WEIGHT
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reasons = []

    # ML reason
    if buy_probability >= MIN_BUY_PROBABILITY:

        reasons.append(
            f"BUY probability is {buy_probability:.1%}"
        )

    elif sell_probability >= MIN_SELL_PROBABILITY:

        reasons.append(
            f"SELL probability is {sell_probability:.1%}"
        )

    else:

        reasons.append(
            "ML model does not show strong directional conviction"
        )

    # Expected return
    if expected_return >= MIN_EXPECTED_RETURN:

        reasons.append(
            f"Expected 20D return is +{expected_return:.2f}%"
        )

    elif expected_return <= MAX_EXPECTED_RETURN_FOR_SELL:

        reasons.append(
            f"Expected 20D return is {expected_return:.2f}%"
        )

    else:

        reasons.append(
            f"Expected 20D return is {expected_return:+.2f}%"
        )

    # Technical reasons
    reasons.extend(
        technical_signal["reasons"]
    )

    # News
    if news_articles > 0:

        reasons.append(
            f"Recent news sentiment is {news_label.lower()} "
            f"across {news_articles} articles"
        )

    # ML quality
    reasons.append(
        f"ML model quality: {overall_quality}"
    )

    # --------------------------------------------------------
    # BUY decision
    # --------------------------------------------------------

    buy_condition = (

        buy_probability >= MIN_BUY_PROBABILITY

        and expected_return >= MIN_EXPECTED_RETURN

        and final_score >= BUY_THRESHOLD
    )

    # --------------------------------------------------------
    # SELL decision
    # --------------------------------------------------------

    sell_condition = (

        sell_probability >= MIN_SELL_PROBABILITY

        and expected_return <= MAX_EXPECTED_RETURN_FOR_SELL

        and final_score <= SELL_THRESHOLD
    )

    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

    if buy_condition:

        signal = "BUY"

    elif sell_condition:

        signal = "SELL"

    else:

        signal = "HOLD"

    # --------------------------------------------------------
    # Position size
    # --------------------------------------------------------

    if signal == "BUY":

        position_size = risk[
            "position_size_percent"
        ]

    else:

        position_size = 0.0

    return {

        "symbol": symbol,

        "price": prediction[
            "price"
        ],

        "buy_probability": buy_probability,

        "sell_probability": sell_probability,

        "expected_return_20d": expected_return,

        "ml_score": ml_score,

        "technical_score": technical_score,

        "news_score": news_score,

        "return_score": return_score,

        "final_score": final_score,

        "ml_quality": overall_quality,

        "news_articles": news_articles,

        "news_label": news_label,

        "signal": signal,

        "position_size_percent": position_size,

        "stop_loss": risk[
            "stop_loss"
        ],

        "take_profit": risk[
            "take_profit"
        ],

        "volatility": risk[
            "volatility"
        ],

        "atr_percent": risk[
            "atr_percent"
        ],

        "reasons": " | ".join(
            reasons
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print("INDIAN STOCK TRADING AI")
    print("TRADING RECOMMENDATION ENGINE")
    print("=" * 120)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    predictions = load_trade_predictions()

    news = load_live_news()

    recommendations = []

    # --------------------------------------------------------
    # Process every stock
    # --------------------------------------------------------

    for _, prediction in predictions.iterrows():

        symbol = prediction[
            "symbol"
        ]

        try:

            # ----------------------------------------------
            # Stock features
            # ----------------------------------------------

            features = load_stock_features(
                symbol
            )

            # ----------------------------------------------
            # Technical signal
            # ----------------------------------------------

            technical_signal = (
                generate_technical_signal(
                    features
                )
            )

            # ----------------------------------------------
            # News
            # ----------------------------------------------

            (
                news_score,
                news_articles,
                news_label
            ) = get_news_score(
                symbol,
                news
            )

            # ----------------------------------------------
            # Risk
            # ----------------------------------------------

            risk = assess_risk(

                features,

                ml_confidence=float(
                    prediction[
                        "effective_buy_probability"
                    ]
                ),

                signal_strength=abs(
                    float(
                        prediction[
                            "effective_buy_probability"
                        ]
                    )
                    - 0.50
                ) * 2,

                entry_price=float(
                    prediction[
                        "price"
                    ]
                )
            )

            # ----------------------------------------------
            # Recommendation
            # ----------------------------------------------

            recommendation = (
                generate_recommendation(

                    prediction,

                    technical_signal,

                    news_score,

                    news_articles,

                    news_label,

                    risk
                )
            )

            recommendations.append(
                recommendation
            )

        except Exception as e:

            print(
                f"\nERROR: {symbol}"
            )

            print(
                str(e)
            )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    if not recommendations:

        raise RuntimeError(
            "No recommendations were generated."
        )

    output = pd.DataFrame(
        recommendations
    )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    output = output.sort_values(
        "final_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 140)
    print("CURRENT STOCK RECOMMENDATIONS")
    print("=" * 140)

    display_columns = [

        "symbol",

        "price",

        "buy_probability",

        "sell_probability",

        "expected_return_20d",

        "ml_quality",

        "technical_score",

        "news_score",

        "final_score",

        "signal",

        "position_size_percent",

        "stop_loss",

        "take_profit",
    ]

    print(
        output[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SIGNAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SIGNAL SUMMARY")
    print("=" * 80)

    print(
        output[
            "signal"
        ].value_counts()
    )

    # ========================================================
    # BUY STOCKS
    # ========================================================

    buy_stocks = output[
        output["signal"] == "BUY"
    ]

    print("\n")
    print("=" * 80)
    print("BUY RECOMMENDATIONS")
    print("=" * 80)

    if buy_stocks.empty:

        print(
            "No BUY recommendations."
        )

    else:

        print(
            buy_stocks[
                [
                    "symbol",
                    "price",
                    "buy_probability",
                    "expected_return_20d",
                    "final_score",
                    "position_size_percent",
                    "stop_loss",
                    "take_profit",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SELL STOCKS
    # ========================================================

    sell_stocks = output[
        output["signal"] == "SELL"
    ]

    print("\n")
    print("=" * 80)
    print("SELL RECOMMENDATIONS")
    print("=" * 80)

    if sell_stocks.empty:

        print(
            "No SELL recommendations."
        )

    else:

        print(
            sell_stocks[
                [
                    "symbol",
                    "price",
                    "sell_probability",
                    "expected_return_20d",
                    "final_score",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # FILE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILE SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()