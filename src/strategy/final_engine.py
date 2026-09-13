from pathlib import Path

import joblib
import pandas as pd

from risk_engine import assess_risk


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT / "data" / "models"
)

NEWS_SENTIMENT_DIR = (
    PROCESSED_DATA_DIR / "news"
)


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


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MIN_ML_CONFIDENCE = 0.50

NEWS_WEIGHT = 0.25
ML_WEIGHT = 0.75


def load_stock_data(symbol):
    """
    Load latest processed technical data.
    """

    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    filepath = (
        PROCESSED_DATA_DIR / filename
    )

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
    Load trained ML model and encoder.
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


def get_ml_prediction(symbol, data):
    """
    Generate the latest ML prediction
    and probability.
    """

    model, encoder = load_model(
        symbol
    )

    latest = data.iloc[-1]

    X = latest[
        FEATURE_COLUMNS
    ].to_frame().T

    prediction_encoded = model.predict(
        X
    )[0]

    prediction = encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    probability_map = {
        class_name: probability
        for class_name, probability
        in zip(
            encoder.classes_,
            probabilities
        )
    }

    confidence = probability_map.get(
        prediction,
        0
    )

    return prediction, confidence


def load_news_sentiment(symbol):
    """
    Load latest company-level news sentiment.
    """

    summary_file = (
        NEWS_SENTIMENT_DIR
        / "company_sentiment_summary.csv"
    )

    if not summary_file.exists():
        return 0.0

    summary = pd.read_csv(
        summary_file
    )

    stock = summary[
        summary["symbol"] == symbol
    ]

    if stock.empty:
        return 0.0

    return float(
        stock.iloc[0][
            "average_sentiment"
        ]
    )


def classify_news_sentiment(score):
    """
    Convert numeric sentiment into a label.
    """

    if score > 0.20:
        return "POSITIVE"

    if score < -0.20:
        return "NEGATIVE"

    return "NEUTRAL"


def calculate_combined_score(
    ml_prediction,
    ml_confidence,
    news_sentiment
):
    """
    Combine ML prediction and news sentiment.

    ML receives 75% weight.
    News receives 25% weight.

    The score ranges approximately
    from -1 to +1.
    """

    if ml_prediction == "UP":

        ml_score = ml_confidence

    else:

        ml_score = -ml_confidence

    news_score = max(
        min(news_sentiment, 1),
        -1
    )

    combined_score = (
        ml_score * ML_WEIGHT
        + news_score * NEWS_WEIGHT
    )

    return combined_score


def generate_signal(
    combined_score
):
    """
    Convert combined score into
    BUY / HOLD / SELL.
    """

    if combined_score >= 0.40:

        return "BUY"

    if combined_score <= -0.40:

        return "SELL"

    return "HOLD"


def calculate_confidence_level(
    combined_score
):
    """
    Human-readable confidence level.
    """

    strength = abs(
        combined_score
    )

    if strength >= 0.70:

        return "HIGH"

    if strength >= 0.50:

        return "MEDIUM"

    return "LOW"


def generate_recommendation(symbol):
    """
    Generate the final AI recommendation.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    data = load_stock_data(
        symbol
    )

    latest_price = float(
        data["Close"].iloc[-1]
    )

    # --------------------------------------------------
    # ML
    # --------------------------------------------------

    (
        ml_prediction,
        ml_confidence
    ) = get_ml_prediction(
        symbol,
        data
    )

    # --------------------------------------------------
    # News
    # --------------------------------------------------

    news_sentiment = (
        load_news_sentiment(
            symbol
        )
    )

    news_label = (
        classify_news_sentiment(
            news_sentiment
        )
    )

    # --------------------------------------------------
    # Combined intelligence
    # --------------------------------------------------

    combined_score = (
        calculate_combined_score(
            ml_prediction,
            ml_confidence,
            news_sentiment
        )
    )

    signal = generate_signal(
        combined_score
    )

    confidence_level = (
        calculate_confidence_level(
            combined_score
        )
    )

    # --------------------------------------------------
    # Risk engine
    # --------------------------------------------------

    risk = assess_risk(
    data=data,
    ml_confidence=ml_confidence,
    signal_strength=combined_score,
    entry_price=latest_price
)

    # --------------------------------------------------
    # Position sizing
    # --------------------------------------------------

    if signal == "BUY":

        position_size = (
            risk["position_size_percent"]
        )

    else:

        position_size = 0.0

    # --------------------------------------------------
    # Reasons
    # --------------------------------------------------

    reasons = []

    if ml_prediction == "UP":

        reasons.append(
            "ML model predicts upward "
            "5-day price direction"
        )

    else:

        reasons.append(
            "ML model predicts downward "
            "5-day price direction"
        )

    if news_label == "POSITIVE":

        reasons.append(
            "Recent news sentiment is positive"
        )

    elif news_label == "NEGATIVE":

        reasons.append(
            "Recent news sentiment is negative"
        )

    else:

        reasons.append(
            "Recent news sentiment is neutral"
        )

    if signal == "BUY":

        reasons.append(
            "Combined ML and news score "
            "supports a BUY signal"
        )

    elif signal == "SELL":

        reasons.append(
            "Combined ML and news score "
            "supports a SELL signal"
        )

    else:

        reasons.append(
            "ML and news signals are not "
            "strong enough for a trade"
        )

    return {
        "symbol": symbol,
        "date": data.index[-1],
        "price": latest_price,
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "news_sentiment": news_sentiment,
        "news_label": news_label,
        "combined_score": combined_score,
        "signal": signal,
        "confidence_level": confidence_level,
        "position_size_percent": position_size,
        "stop_loss": risk["stop_loss"],
        "take_profit": risk["take_profit"],
        "volatility": risk["volatility"],
        "atr_percent": risk["atr_percent"],
        "reasons": " | ".join(reasons),
    }


def main():

    print("\n" + "=" * 100)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "FINAL AI DECISION ENGINE"
    )
    print("=" * 100)

    model_directories = [
        path
        for path in MODEL_DIR.iterdir()
        if path.is_dir()
        and (path / "model.pkl").exists()
    ]

    recommendations = []

    for model_dir in sorted(
        model_directories
    ):

        symbol = (
            model_dir.name
            .replace("_", ".")
        )

        try:

            result = (
                generate_recommendation(
                    symbol
                )
            )

            recommendations.append(
                result
            )

            print("\n" + "-" * 80)

            print(
                f"{symbol}"
            )

            print(
                f"Price: "
                f"₹{result['price']:.2f}"
            )

            print(
                f"ML: "
                f"{result['ml_prediction']} "
                f"({result['ml_confidence']:.2%})"
            )

            print(
                f"News: "
                f"{result['news_label']} "
                f"({result['news_sentiment']:+.2f})"
            )

            print(
                f"Combined Score: "
                f"{result['combined_score']:+.3f}"
            )

            print(
                f"FINAL SIGNAL: "
                f"{result['signal']}"
            )

            print(
                f"Confidence Level: "
                f"{result['confidence_level']}"
            )

            print(
                f"Position Size: "
                f"{result['position_size_percent']:.2f}%"
            )

            print(
                f"Stop Loss: "
                f"₹{result['stop_loss']:.2f}"
            )

            print(
                f"Take Profit: "
                f"₹{result['take_profit']:.2f}"
            )

            print(
                "Reasons:"
            )

            for reason in result[
                "reasons"
            ].split(" | "):

                print(
                    f"  • {reason}"
                )

        except Exception as e:

            print(
                f"\n{symbol} ERROR: {e}"
            )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    if recommendations:

        results = pd.DataFrame(
            recommendations
        )

        results.sort_values(
            "combined_score",
            ascending=False,
            inplace=True
        )

        output_file = (
            MODEL_DIR
            / "final_recommendations.csv"
        )

        results.to_csv(
            output_file,
            index=False
        )

        print("\n" + "=" * 100)
        print(
            "FINAL ENGINE COMPLETE"
        )
        print("=" * 100)

        print(
            results[
                [
                    "symbol",
                    "price",
                    "ml_prediction",
                    "ml_confidence",
                    "news_label",
                    "news_sentiment",
                    "combined_score",
                    "signal",
                    "position_size_percent",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            f"\nSaved to:\n"
            f"{output_file}"
        )


if __name__ == "__main__":
    main()