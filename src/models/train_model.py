import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "RELIANCE_NS"
)

MODEL_DIR = PROJECT_ROOT / "data" / "models"


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


def load_training_data():

    file_path = TRAINING_DATA_DIR / "training_data.csv"

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    # Make sure the data is in chronological order
    data.sort_index(inplace=True)

    return data


def prepare_data(data):

    X = data[FEATURE_COLUMNS]

    y = data["Target"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder


def train_model(X, y, encoder):

    # ---------------------------------
    # Chronological train/test split
    # ---------------------------------

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    print("\nTraining rows:", len(X_train))

    print("Testing rows:", len(X_test))

    print("\nTraining date range:")

    print(X_train.index.min())

    print("to")

    print(X_train.index.max())

    print("\nTesting date range:")

    print(X_test.index.min())

    print("to")

    print(X_test.index.max())

    # ---------------------------------
    # Random Forest
    # ---------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Accuracy:")

    print(f"{accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    return model


def save_model(model, encoder):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / "reliance_model.pkl"

    encoder_path = MODEL_DIR / "label_encoder.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        encoder,
        encoder_path
    )

    print("\nModel saved to:")

    print(model_path)

    print("\nEncoder saved to:")

    print(encoder_path)


def main():

    print("\nLoading training data...")

    data = load_training_data()

    print(
        f"Total rows: {len(data)}"
    )

    X, y, encoder = prepare_data(data)

    print(
        f"Features used: {X.shape[1]}"
    )

    model = train_model(
        X,
        y,
        encoder
    )

    save_model(
        model,
        encoder
    )


if __name__ == "__main__":
    main()