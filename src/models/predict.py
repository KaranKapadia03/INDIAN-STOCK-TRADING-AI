import pandas as pd
from pathlib import Path
import joblib


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


def load_latest_data():

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


def predict():

    model, encoder = load_model()

    data = load_latest_data()

    latest_data = data.iloc[[-1]]

    X = latest_data[FEATURE_COLUMNS]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    signal = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = probabilities[prediction]

    latest_price = latest_data["Close"].iloc[0]

    latest_rsi = latest_data["RSI_14"].iloc[0]

    latest_macd = latest_data["MACD"].iloc[0]

    latest_signal = latest_data["MACD_Signal"].iloc[0]

    print("\n" + "=" * 50)

    print("INDIAN STOCK TRADING AI")

    print("=" * 50)

    print("\nStock: RELIANCE")

    print(
        f"Latest Price: ₹{latest_price:.2f}"
    )

    print(
        f"RSI: {latest_rsi:.2f}"
    )

    print(
        f"MACD: {latest_macd:.2f}"
    )

    print(
        f"MACD Signal: {latest_signal:.2f}"
    )

    print("\nAI Prediction:")

    print(
        f"Signal: {signal}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    print("\nClass probabilities:")

    for class_name, probability in zip(
        encoder.classes_,
        probabilities
    ):

        print(
            f"{class_name}: {probability:.2%}"
        )


if __name__ == "__main__":

    predict()