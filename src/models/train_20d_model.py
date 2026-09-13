from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

HORIZON = 20

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
]


def load_data(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features.csv"
    )

    df = pd.read_csv(path)

    date_candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
    ]

    date_column = None

    for column in date_candidates:
        if column in df.columns:
            date_column = column
            break

    if date_column is None:
        date_column = df.columns[0]

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df.dropna(
        subset=["date"],
        inplace=True
    )

    df.sort_values(
        "date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


def get_feature_columns(df):

    excluded = {
        "date",
        "Future_Return",
        "Target",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        # Do not use current/live news in the
        # historical model.
        if column.startswith("News_"):
            continue

        if column == "Days_Since_News":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(column)

    return features


def prepare_data(df, features):

    df = df.copy()

    # --------------------------------------------------------
    # 20-day future return
    # --------------------------------------------------------

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # --------------------------------------------------------
    # Classification target
    #
    # UP   = positive 20-day return
    # DOWN = zero or negative return
    # --------------------------------------------------------

    df["Target"] = np.where(
        df["Future_Return_20D"] > 0,
        "UP",
        "DOWN"
    )

    df.dropna(
        subset=features + [
            "Future_Return_20D"
        ],
        inplace=True
    )

    return df


def train_stock(symbol):

    print("\n" + "=" * 70)
    print(f"Training 20-Day Model: {symbol}")
    print("=" * 70)

    df = load_data(symbol)

    features = get_feature_columns(df)

    df = prepare_data(
        df,
        features
    )

    # --------------------------------------------------------
    # Chronological 80/20 split
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    X_train = train[features]
    X_test = test[features]

    y_train = train["Target"]
    y_test = test["Target"]

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train_encoded
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # Find UP class
    up_class = label_encoder.transform(
        ["UP"]
    )[0]

    up_index = list(
        model.classes_
    ).index(up_class)

    up_probability = probabilities[
        :,
        up_index
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_encoded,
        predictions
    )

    try:
        auc = roc_auc_score(
            y_test_encoded,
            up_probability
        )
    except ValueError:
        auc = np.nan

    print(
        f"Train observations: {len(train)}"
    )

    print(
        f"Test observations:  {len(test)}"
    )

    print(
        f"Features:             {len(features)}"
    )

    print(
        f"Accuracy:             {accuracy:.4f}"
    )

    print(
        f"ROC-AUC:              {auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_encoded,
            predictions,
            target_names=
            label_encoder.classes_,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "date": test["date"].values,

            "symbol": symbol,

            "Close": test[
                "Close"
            ].values,

            "Future_Return_20D":
                test[
                    "Future_Return_20D"
                ].values,

            "Actual":
                y_test.values,

            "Prediction":
                label_encoder.inverse_transform(
                    predictions
                ),

            "UP_Probability":
                up_probability,
        }
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_dir = (
        MODELS_DIR
        / f"{symbol.replace('.', '_')}_20D"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_dir / "model.pkl"
    )

    joblib.dump(
        label_encoder,
        model_dir / "label_encoder.pkl"
    )

    # Save feature list
    joblib.dump(
        features,
        model_dir / "features.pkl"
    )

    prediction_df.to_csv(
        model_dir / "test_predictions.csv",
        index=False
    )

    return {
        "symbol": symbol,
        "train_size": len(train),
        "test_size": len(test),
        "features": len(features),
        "accuracy": accuracy,
        "auc": auc,
    }


def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "20-DAY ML MODEL TRAINING"
    )
    print("=" * 70)

    results = []

    for symbol in STOCKS:

        try:

            result = train_stock(
                symbol
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                f"\nERROR: {symbol}"
            )

            print(e)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print(
            "\nNo models were trained."
        )

        return

    summary_path = (
        MODELS_DIR
        / "20d_model_results.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        "20-DAY MODEL SUMMARY"
    )
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Average Accuracy: "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print(
        f"Average ROC-AUC: "
        f"{results_df['auc'].mean():.4f}"
    )

    print("\nModels saved to:")

    print(
        MODELS_DIR
    )


if __name__ == "__main__":
    main()