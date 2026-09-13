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
    Load the latest processed stock data.
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


def get_ml_prediction(symbol, data):
    """
    Generate the latest ML prediction
    and confidence.
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


def convert_prediction_to_signal(
    prediction,
    confidence
):
    """
    Convert ML prediction into
    BUY / HOLD / SELL.

    The current model predicts
    UP or DOWN, so we use confidence
    to decide whether the signal is
    strong enough.
    """

    if confidence < 0.50:
        return "HOLD"

    if prediction == "UP":
        return "BUY"

    return "SELL"


def generate_recommendation(symbol):
    """
    Generate complete recommendation
    for one stock.
    """

    data = load_stock_data(
        symbol
    )

    prediction, confidence = (
        get_ml_prediction(
            symbol,
            data
        )
    )

    signal = convert_prediction_to_signal(
        prediction,
        confidence
    )

    latest_price = data[
        "Close"
    ].iloc[-1]

    # --------------------------------------------------
    # Risk assessment
    # --------------------------------------------------

    risk = assess_risk(
        data=data,
        confidence=confidence,
        entry_price=latest_price
    )

    # --------------------------------------------------
    # Risk-aware recommendation
    # --------------------------------------------------

    if signal == "BUY":

        position_size = (
            risk["position_size_percent"]
        )

        if position_size <= 0:

            final_signal = "HOLD"

        else:

            final_signal = "BUY"

    elif signal == "SELL":

        final_signal = "SELL"

        position_size = 0

    else:

        final_signal = "HOLD"

        position_size = 0

    # --------------------------------------------------
    # Confidence level
    # --------------------------------------------------

    if confidence >= 0.75:

        confidence_level = "HIGH"

    elif confidence >= 0.60:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

    return {
        "symbol": symbol,
        "date": data.index[-1],
        "price": latest_price,
        "ml_prediction": prediction,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "signal": final_signal,
        "position_size_percent":
            position_size,
        "stop_loss": risk["stop_loss"],
        "take_profit":
            risk["take_profit"],
        "volatility":
            risk["volatility"],
        "atr_percent":
            risk["atr_percent"],
    }


def main():

    print("\n" + "=" * 90)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "AI RECOMMENDATION ENGINE V2"
    )
    print("=" * 90)

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

            recommendation = (
                generate_recommendation(
                    symbol
                )
            )

            recommendations.append(
                recommendation
            )

            print(
                f"\n{symbol}"
            )

            print(
                f"Price: "
                f"₹{recommendation['price']:.2f}"
            )

            print(
                f"ML Prediction: "
                f"{recommendation['ml_prediction']}"
            )

            print(
                f"Confidence: "
                f"{recommendation['confidence']:.2%}"
            )

            print(
                f"Signal: "
                f"{recommendation['signal']}"
            )

            print(
                f"Position Size: "
                f"{recommendation['position_size_percent']:.2f}%"
            )

            print(
                f"Stop Loss: "
                f"₹{recommendation['stop_loss']:.2f}"
            )

            print(
                f"Take Profit: "
                f"₹{recommendation['take_profit']:.2f}"
            )

        except Exception as e:

            print(
                f"\n{symbol} ERROR: {e}"
            )

    # --------------------------------------------------
    # Save recommendations
    # --------------------------------------------------

    if recommendations:

        results = pd.DataFrame(
            recommendations
        )

        results.sort_values(
            "confidence",
            ascending=False,
            inplace=True
        )

        output_file = (
            MODEL_DIR
            / "recommendations_v2.csv"
        )

        results.to_csv(
            output_file,
            index=False
        )

        print("\n" + "=" * 90)
        print(
            "RECOMMENDATION ENGINE COMPLETE"
        )
        print("=" * 90)

        print(
            results[
                [
                    "symbol",
                    "price",
                    "ml_prediction",
                    "confidence",
                    "signal",
                    "position_size_percent",
                    "stop_loss",
                    "take_profit",
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