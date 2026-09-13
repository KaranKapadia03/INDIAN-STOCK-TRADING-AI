import pandas as pd
from pathlib import Path
import joblib

from technical_signal import generate_technical_signal


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "models"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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


def load_model():

    model = joblib.load(
        MODEL_DIR / "reliance_model.pkl"
    )

    encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    return model, encoder


def load_data():

    file_path = (
        PROCESSED_DATA_DIR
        / "RELIANCE_NS_features.csv"
    )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(inplace=True)

    return data


def get_ml_prediction(data):

    model, encoder = load_model()

    latest = data.iloc[[-1]]

    X = latest[FEATURE_COLUMNS]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    signal = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = probabilities[prediction]

    return signal, confidence


def combine_signals(ml_signal, ml_confidence, technical_signal):

    score = 0

    # -----------------------------
    # ML signal
    # -----------------------------

    if ml_signal == "BUY":
        score += 2

    elif ml_signal == "SELL":
        score -= 2

    # -----------------------------
    # Technical signal
    # -----------------------------

    if technical_signal == "BUY":
        score += 1

    elif technical_signal == "SELL":
        score -= 1

    # -----------------------------
    # Confidence adjustment
    # -----------------------------

    if ml_confidence < 0.50:

        # Weak ML confidence
        score = int(score * 0.5)

    # -----------------------------
    # Final decision
    # -----------------------------

    if score >= 2:
        final_signal = "BUY"

    elif score <= -2:
        final_signal = "SELL"

    else:
        final_signal = "HOLD"

    return final_signal, score


def main():

    print("\n" + "=" * 60)

    print("INDIAN STOCK TRADING AI")

    print("=" * 60)

    data = load_data()

    ml_signal, ml_confidence = get_ml_prediction(
        data
    )

    technical_result = generate_technical_signal(
        data
    )

    technical_signal = technical_result["signal"]

    final_signal, final_score = combine_signals(
        ml_signal,
        ml_confidence,
        technical_signal
    )

    latest_price = data["Close"].iloc[-1]

    print(
        f"\nStock: RELIANCE"
    )

    print(
        f"Latest Price: ₹{latest_price:.2f}"
    )

    print("\nML Model")

    print(
        f"Signal: {ml_signal}"
    )

    print(
        f"Confidence: {ml_confidence:.2%}"
    )

    print("\nTechnical Analysis")

    print(
        f"Signal: {technical_signal}"
    )

    print(
        f"Score: {technical_result['score']}"
    )

    print("\nFinal AI Decision")

    print(
        f"Signal: {final_signal}"
    )

    print(
        f"Combined Score: {final_score}"
    )

    print("\nTechnical Reasons:")

    for reason in technical_result["reasons"]:

        print(
            f"- {reason}"
        )


if __name__ == "__main__":

    main()