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


def load_stock_data(symbol):
    """
    Load the latest processed feature data for a stock.
    """

    filename = (
        symbol.replace(".", "_")
        + "_features.csv"
    )

    file_path = (
        PROCESSED_DATA_DIR / filename
    )

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
    """
    Load the trained model and label encoder
    for a stock.
    """

    stock_model_dir = (
        MODEL_DIR
        / symbol.replace(".", "_")
    )

    model_path = (
        stock_model_dir / "model.pkl"
    )

    encoder_path = (
        stock_model_dir / "label_encoder.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for {symbol}"
        )

    model = joblib.load(
        model_path
    )

    encoder = joblib.load(
        encoder_path
    )

    return model, encoder


def predict_stock(symbol):
    """
    Generate an ML prediction for one stock.
    """

    data = load_stock_data(symbol)

    model, encoder = load_stock_model(
        symbol
    )

    latest_data = data.iloc[[-1]]

    X = latest_data[FEATURE_COLUMNS]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    signal = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = probabilities[
        prediction
    ]

    latest_price = (
        latest_data["Close"].iloc[0]
    )

    return {
        "symbol": symbol,
        "price": latest_price,
        "signal": signal,
        "confidence": confidence,
        "buy_probability": probabilities[
            list(encoder.classes_).index("BUY")
        ] if "BUY" in encoder.classes_ else 0,
        "hold_probability": probabilities[
            list(encoder.classes_).index("HOLD")
        ] if "HOLD" in encoder.classes_ else 0,
        "sell_probability": probabilities[
            list(encoder.classes_).index("SELL")
        ] if "SELL" in encoder.classes_ else 0,
    }


def main():

    print("\n" + "=" * 70)

    print("INDIAN STOCK TRADING AI")

    print("ML PREDICTIONS FOR ALL STOCKS")

    print("=" * 70)

    feature_files = sorted(
        PROCESSED_DATA_DIR.glob(
            "*_features.csv"
        )
    )

    results = []

    for file_path in feature_files:

        symbol = (
            file_path.stem
            .replace("_features", "")
            .replace("_", ".")
        )

        try:

            result = predict_stock(
                symbol
            )

            results.append(result)

            print(
                f"\n{symbol}"
            )

            print(
                f"Price: ₹"
                f"{result['price']:.2f}"
            )

            print(
                f"Signal: "
                f"{result['signal']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2%}"
            )

        except Exception as e:

            print(
                f"\nERROR - {symbol}: {e}"
            )

    # --------------------------------
    # Create summary
    # --------------------------------

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df.sort_values(
            "confidence",
            ascending=False,
            inplace=True
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "PREDICTION SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            results_df[
                [
                    "symbol",
                    "price",
                    "signal",
                    "confidence",
                    "buy_probability",
                    "hold_probability",
                    "sell_probability",
                ]
            ].to_string(
                index=False
            )
        )

        output_file = (
            MODEL_DIR
            / "latest_predictions.csv"
        )

        results_df.to_csv(
            output_file,
            index=False
        )

        print(
            "\nPredictions saved to:"
        )

        print(output_file)


if __name__ == "__main__":

    main()