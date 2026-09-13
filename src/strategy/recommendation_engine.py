import sys
from pathlib import Path

import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

STRATEGY_DIR = Path(__file__).resolve().parent
DATA_SRC_DIR = PROJECT_ROOT / "src" / "data"

sys.path.append(str(STRATEGY_DIR))
sys.path.append(str(DATA_SRC_DIR))

from technical_signal import generate_technical_signal
from stock_universe import get_stock_universe


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
]


def load_stock_data(symbol):
    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(inplace=True)

    return data


def load_stock_model(symbol):
    stock_model_dir = (
        MODEL_DIR
        / symbol.replace(".", "_")
    )

    model_path = stock_model_dir / "model.pkl"
    encoder_path = stock_model_dir / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    return model, encoder


def get_ml_prediction(data, symbol):

    model, encoder = load_stock_model(symbol)

    latest_data = data.iloc[[-1]]

    X = latest_data[FEATURE_COLUMNS]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    signal = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = probabilities[prediction]

    return signal, confidence


def generate_recommendation(
    ml_signal,
    ml_confidence,
    technical_signal,
    technical_score
):

    if ml_confidence >= 0.65:
        confidence_level = "HIGH"

    elif ml_confidence >= 0.50:
        confidence_level = "MEDIUM"

    else:
        confidence_level = "LOW"

    signals_agree = (
        ml_signal == technical_signal
    )

    if ml_confidence < 0.40:

        recommendation = "HOLD"

        reason = (
            "ML confidence is too low "
            "for a directional recommendation."
        )

    elif signals_agree:

        if ml_signal == "BUY":

            if ml_confidence >= 0.65:
                recommendation = "STRONG BUY"
            else:
                recommendation = "BUY"

            reason = (
                "ML and technical analysis "
                "both indicate bullish conditions."
            )

        elif ml_signal == "SELL":

            if ml_confidence >= 0.65:
                recommendation = "STRONG SELL"
            else:
                recommendation = "SELL"

            reason = (
                "ML and technical analysis "
                "both indicate bearish conditions."
            )

        else:

            recommendation = "HOLD"

            reason = (
                "ML and technical analysis "
                "both indicate neutral conditions."
            )

    else:

        recommendation = "HOLD"

        reason = (
            "ML and technical analysis disagree, "
            "so the system avoids a directional trade."
        )

    return {
        "recommendation": recommendation,
        "confidence_level": confidence_level,
        "signals_agree": signals_agree,
        "reason": reason,
    }


def analyze_stock(symbol):

    data = load_stock_data(symbol)

    ml_signal, ml_confidence = get_ml_prediction(
        data,
        symbol
    )

    technical_result = generate_technical_signal(
        data
    )

    recommendation = generate_recommendation(
        ml_signal=ml_signal,
        ml_confidence=ml_confidence,
        technical_signal=technical_result["signal"],
        technical_score=technical_result["score"]
    )

    latest = data.iloc[-1]

    return {
        "symbol": symbol,
        "date": latest.name,
        "price": latest["Close"],
        "ml_signal": ml_signal,
        "ml_confidence": ml_confidence,
        "technical_signal": technical_result["signal"],
        "technical_score": technical_result["score"],
        "recommendation": recommendation["recommendation"],
        "confidence_level": recommendation["confidence_level"],
        "signals_agree": recommendation["signals_agree"],
        "reason": recommendation["reason"],
    }


def main():

    print("\n" + "=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("ALL STOCK RECOMMENDATIONS")
    print("=" * 80)

    stocks = get_stock_universe()

    results = []

    for symbol in stocks["symbol"]:

        print(f"\nAnalyzing {symbol}...")

        try:

            result = analyze_stock(symbol)

            results.append(result)

            print(
                f"Price: ₹{result['price']:.2f} | "
                f"ML: {result['ml_signal']} "
                f"({result['ml_confidence']:.2%}) | "
                f"Technical: {result['technical_signal']} | "
                f"Final: {result['recommendation']}"
            )

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

    if not results:
        print("\nNo recommendations generated.")
        return

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        "ml_confidence",
        ascending=False,
        inplace=True
    )

    print("\n" + "=" * 100)
    print("FINAL STOCK RECOMMENDATIONS")
    print("=" * 100)

    print(
        results_df[
            [
                "symbol",
                "price",
                "ml_signal",
                "ml_confidence",
                "technical_signal",
                "technical_score",
                "recommendation",
                "confidence_level",
                "signals_agree",
            ]
        ].to_string(index=False)
    )

    output_file = (
        MODEL_DIR
        / "all_recommendations.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nRecommendations saved to:")
    print(output_file)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "\nRecommendation counts:"
    )

    print(
        results_df[
            "recommendation"
        ].value_counts()
    )


if __name__ == "__main__":
    main()