from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "walk_forward_20d"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

INITIAL_TRAIN_RATIO = 0.60

RETRAIN_DAYS = 60

TEST_WINDOW_DAYS = 60

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


# ============================================================
# LOAD DATA
# ============================================================

def load_stock(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
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

        first_column = df.columns[0]

        parsed = pd.to_datetime(
            df[first_column],
            errors="coerce"
        )

        if parsed.notna().mean() > 0.90:

            date_column = first_column

    if date_column is None:

        raise ValueError(
            f"No date column found in {path}"
        )

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


# ============================================================
# FEATURES
# ============================================================

def get_features(df):

    excluded = {
        "date",
        "Future_Return",
        "Future_Return_20D",
        "Target",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        # Historical news is not available reliably,
        # therefore exclude all news features.
        if column.startswith("News_"):
            continue

        if column == "Days_Since_News":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(column)

    return features


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_data(df, features):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

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

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# WALK-FORWARD MODEL
# ============================================================

def walk_forward_stock(symbol):

    print("\n" + "=" * 75)
    print(
        f"20-DAY WALK-FORWARD: {symbol}"
    )
    print("=" * 75)

    df = load_stock(symbol)

    features = get_features(df)

    df = prepare_data(
        df,
        features
    )

    total_rows = len(df)

    initial_train_size = int(
        total_rows * INITIAL_TRAIN_RATIO
    )

    predictions = []

    train_end = initial_train_size

    window_number = 0

    while train_end < total_rows:

        test_start = train_end

        test_end = min(
            test_start + TEST_WINDOW_DAYS,
            total_rows
        )

        train = df.iloc[
            :train_end
        ]

        test = df.iloc[
            test_start:test_end
        ]

        if len(test) == 0:

            break

        X_train = train[
            features
        ]

        y_train = train[
            "Target"
        ]

        X_test = test[
            features
        ]

        y_test = test[
            "Target"
        ]

        # ----------------------------------------------------
        # Encode target
        # ----------------------------------------------------

        encoder = LabelEncoder()

        y_train_encoded = (
            encoder.fit_transform(
                y_train
            )
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=200,
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

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        up_class = encoder.transform(
            ["UP"]
        )[0]

        up_index = list(
            model.classes_
        ).index(up_class)

        up_probability = probabilities[
            :,
            up_index
        ]

        predicted_labels = (
            encoder.inverse_transform(
                predicted
            )
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        fold = pd.DataFrame(
            {
                "date":
                    test["date"].values,

                "symbol":
                    symbol,

                "Close":
                    test["Close"].values,

                "Future_Return_20D":
                    test[
                        "Future_Return_20D"
                    ].values,

                "Actual":
                    y_test.values,

                "Prediction":
                    predicted_labels,

                "UP_Probability":
                    up_probability,

                "Fold":
                    window_number,

                "Train_End_Date":
                    train["date"].iloc[-1],
            }
        )

        predictions.append(
            fold
        )

        print(
            f"Fold {window_number + 1}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"{test['date'].iloc[0].date()} "
            f"to "
            f"{test['date'].iloc[-1].date()}"
        )

        window_number += 1

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        train_end += RETRAIN_DAYS

    if not predictions:

        return None

    predictions_df = pd.concat(
        predictions,
        ignore_index=True
    )

    # Remove accidental duplicate prediction dates
    predictions_df.drop_duplicates(
        subset=["date"],
        keep="first",
        inplace=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    actual = predictions_df[
        "Actual"
    ]

    predicted = predictions_df[
        "Prediction"
    ]

    probabilities = predictions_df[
        "UP_Probability"
    ]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    actual_binary = (
        actual == "UP"
    ).astype(int)

    try:

        auc = roc_auc_score(
            actual_binary,
            probabilities
        )

    except ValueError:

        auc = np.nan

    # --------------------------------------------------------
    # Probability / return correlation
    # --------------------------------------------------------

    probability_return_corr = (
        predictions_df[
            "UP_Probability"
        ].corr(
            predictions_df[
                "Future_Return_20D"
            ]
        )
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    up_rate = (
        actual_binary.mean()
    )

    baseline_accuracy = max(
        up_rate,
        1 - up_rate
    )

    # --------------------------------------------------------
    # High-confidence performance
    # --------------------------------------------------------

    threshold_results = []

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        subset = predictions_df[
            predictions_df[
                "UP_Probability"
            ] >= threshold
        ]

        if subset.empty:

            continue

        threshold_results.append(
            {
                "symbol":
                    symbol,

                "threshold":
                    threshold,

                "observations":
                    len(subset),

                "average_return_pct":
                    subset[
                        "Future_Return_20D"
                    ].mean(),

                "median_return_pct":
                    subset[
                        "Future_Return_20D"
                    ].median(),

                "win_rate_pct":
                    (
                        subset[
                            "Future_Return_20D"
                        ]
                        .gt(0)
                        .mean()
                        * 100
                    ),

                "average_probability":
                    subset[
                        "UP_Probability"
                    ].mean(),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    # --------------------------------------------------------
    # Save threshold results
    # --------------------------------------------------------

    threshold_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_file,
        index=False
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"Probability/Return Correlation: "
        f"{probability_return_corr:.4f}"
    )

    print(
        f"Predictions: "
        f"{len(predictions_df)}"
    )

    return {
        "symbol":
            symbol,

        "predictions":
            len(predictions_df),

        "accuracy":
            accuracy,

        "baseline_accuracy":
            baseline_accuracy,

        "roc_auc":
            auc,

        "probability_return_correlation":
            probability_return_corr,

        "actual_up_rate":
            up_rate,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "20-DAY WALK-FORWARD VALIDATION"
    )
    print("=" * 75)

    results = []

    for symbol in STOCKS:

        try:

            result = walk_forward_stock(
                symbol
            )

            if result is not None:

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

    summary = pd.DataFrame(
        results
    )

    if summary.empty:

        print(
            "\nNo results generated."
        )

        return

    summary_file = (
        OUTPUT_DIR
        / "walk_forward_20d_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    print("=" * 75)
    print(
        "WALK-FORWARD 20-DAY SUMMARY"
    )
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Average Accuracy: "
        f"{summary['accuracy'].mean():.4f}"
    )

    print(
        f"Average Baseline: "
        f"{summary['baseline_accuracy'].mean():.4f}"
    )

    print(
        f"Average ROC-AUC: "
        f"{summary['roc_auc'].mean():.4f}"
    )

    print(
        f"Average Probability/Return "
        f"Correlation: "
        f"{summary['probability_return_correlation'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Best stocks
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(
        "BEST 20-DAY MODELS BY ROC-AUC"
    )
    print("=" * 75)

    print(
        summary.sort_values(
            "roc_auc",
            ascending=False
        )[
            [
                "symbol",
                "roc_auc",
                "accuracy",
                "baseline_accuracy",
                "probability_return_correlation",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Weak models
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(
        "WEAKEST 20-DAY MODELS"
    )
    print("=" * 75)

    print(
        summary.sort_values(
            "roc_auc",
            ascending=True
        )[
            [
                "symbol",
                "roc_auc",
                "accuracy",
                "baseline_accuracy",
                "probability_return_correlation",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 75)
    print(
        "20-DAY WALK-FORWARD VALIDATION COMPLETE"
    )
    print("=" * 75)

    print(
        f"\nResults saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()