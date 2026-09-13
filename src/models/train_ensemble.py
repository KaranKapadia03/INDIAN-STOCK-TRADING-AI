from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "ensemble"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PARAMETERS
# =========================================================

TRAIN_SIZE = 0.60

TEST_WINDOW = 60

RETRAIN_EVERY = 60

BUY_THRESHOLD = 0.60


# =========================================================
# ENSEMBLE WEIGHTS
# =========================================================

RF_WEIGHT = 0.40

GB_WEIGHT = 0.40

LR_WEIGHT = 0.20


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
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

    # NIFTY
    "NIFTY_Close",
    "NIFTY_Return_1D",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "NIFTY_SMA_20",
    "NIFTY_SMA_50",
    "NIFTY_Volatility_20D",
    "NIFTY_Price_vs_SMA20",
    "NIFTY_Price_vs_SMA50",

    # BANK NIFTY
    "BANKNIFTY_Close",
    "BANKNIFTY_Return_1D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
    "BANKNIFTY_SMA_20",
    "BANKNIFTY_SMA_50",
    "BANKNIFTY_Volatility_20D",
    "BANKNIFTY_Price_vs_SMA20",
    "BANKNIFTY_Price_vs_SMA50",

    # Relative strength
    "Relative_5D_vs_NIFTY",
    "Relative_20D_vs_NIFTY",
    "Relative_5D_vs_BANKNIFTY",
    "Relative_20D_vs_BANKNIFTY",
    "Relative_Volatility_NIFTY",
    "Relative_Volatility_BANKNIFTY",
    "Relative_Price_vs_NIFTY",
    "Relative_Price_vs_BANKNIFTY",
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


# =========================================================
# TARGET
# =========================================================

def create_target(data):

    data = data.copy()

    data["Future_Return_5D"] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    # 1 = future 5-day return > +1%
    # 0 = otherwise

    data["Target"] = (
        data["Future_Return_5D"] > 1
    ).astype(int)

    data.dropna(
        subset=[
            "Future_Return_5D"
        ],
        inplace=True
    )

    return data


# =========================================================
# CREATE MODELS
# =========================================================

def create_models():

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
    )

    logistic_regression = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ]
    )

    return (
        random_forest,
        gradient_boosting,
        logistic_regression,
    )


# =========================================================
# WALK-FORWARD
# =========================================================

def walk_forward(data):

    predictions = []

    n = len(data)

    train_end = int(
        n * TRAIN_SIZE
    )

    while train_end < n:

        test_end = min(
            train_end + TEST_WINDOW,
            n
        )

        train_data = data.iloc[
            :train_end
        ]

        test_data = data.iloc[
            train_end:test_end
        ]

        if test_data.empty:
            break

        X_train = train_data[
            FEATURES
        ]

        y_train = train_data[
            "Target"
        ]

        X_test = test_data[
            FEATURES
        ]

        # -------------------------------------------------
        # CREATE MODELS
        # -------------------------------------------------

        (
            rf,
            gb,
            lr
        ) = create_models()

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        rf.fit(
            X_train,
            y_train
        )

        gb.fit(
            X_train,
            y_train
        )

        lr.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        rf_probability = (
            rf.predict_proba(
                X_test
            )[:, 1]
        )

        gb_probability = (
            gb.predict_proba(
                X_test
            )[:, 1]
        )

        lr_probability = (
            lr.predict_proba(
                X_test
            )[:, 1]
        )

        # -------------------------------------------------
        # ENSEMBLE
        # -------------------------------------------------

        ensemble_probability = (
            rf_probability * RF_WEIGHT
            +
            gb_probability * GB_WEIGHT
            +
            lr_probability * LR_WEIGHT
        )

        # -------------------------------------------------
        # SAVE PREDICTIONS
        # -------------------------------------------------

        for j, date in enumerate(
            test_data.index
        ):

            predictions.append(
                {
                    "date": date,

                    "RF_Probability":
                        rf_probability[j],

                    "GB_Probability":
                        gb_probability[j],

                    "LR_Probability":
                        lr_probability[j],

                    "Ensemble_Probability":
                        ensemble_probability[j],

                    "Actual_Target":
                        test_data.iloc[j][
                            "Target"
                        ],

                    "Future_Return_5D":
                        test_data.iloc[j][
                            "Future_Return_5D"
                        ],
                }
            )

        # Move forward
        train_end += RETRAIN_EVERY

    return pd.DataFrame(
        predictions
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_predictions(
    predictions
):

    y_true = predictions[
        "Actual_Target"
    ]

    y_probability = predictions[
        "Ensemble_Probability"
    ]

    y_pred = (
        y_probability
        >= BUY_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_true,
            y_probability
        )

    except ValueError:

        auc = np.nan

    positive_rate = (
        y_true.mean()
    )

    predicted_positive_rate = (
        y_pred.mean()
    )

    baseline_accuracy = max(
        positive_rate,
        1 - positive_rate
    )

    return {
        "accuracy":
            accuracy * 100,

        "precision":
            precision * 100,

        "recall":
            recall * 100,

        "f1":
            f1 * 100,

        "roc_auc":
            auc,

        "baseline_accuracy":
            baseline_accuracy * 100,

        "actual_positive_rate":
            positive_rate * 100,

        "predicted_positive_rate":
            predicted_positive_rate * 100,
    }


# =========================================================
# PROCESS STOCK
# =========================================================

def process_stock(filepath):

    symbol = (
        filepath.stem
        .replace(
            "_features",
            ""
        )
        .replace(
            "_",
            "."
        )
    )

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data.sort_index(
        inplace=True
    )

    data.dropna(
        subset=FEATURES,
        inplace=True
    )

    data = create_target(
        data
    )

    predictions = walk_forward(
        data
    )

    if predictions.empty:
        return None

    metrics = evaluate_predictions(
        predictions
    )

    metrics["symbol"] = symbol

    predictions.to_csv(
        OUTPUT_DIR
        / (
            symbol.replace(
                ".",
                "_"
            )
            + "_predictions.csv"
        ),
        index=False
    )

    return metrics


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("WALK-FORWARD ML ENSEMBLE")
    print("=" * 80)

    print("\nModels:")

    print(
        "Random Forest: 40%"
    )

    print(
        "Gradient Boosting: 40%"
    )

    print(
        "Logistic Regression: 20%"
    )

    print(
        f"\nBuy threshold: "
        f"{BUY_THRESHOLD * 100:.0f}%"
    )

    print("\nTraining:")

    print(
        "60% initial training data"
    )

    print(
        "60 trading-day test windows"
    )

    print(
        "Retrain every 60 trading days"
    )

    files = sorted(
        FEATURES_DIR.glob(
            "*_features.csv"
        )
    )

    all_metrics = []

    for filepath in files:

        try:

            print(
                f"\nProcessing "
                f"{filepath.name}..."
            )

            metrics = process_stock(
                filepath
            )

            if metrics is None:
                continue

            all_metrics.append(
                metrics
            )

            print(
                f"AUC: "
                f"{metrics['roc_auc']:.3f} | "
                f"Accuracy: "
                f"{metrics['accuracy']:.2f}% | "
                f"Precision: "
                f"{metrics['precision']:.2f}% | "
                f"Recall: "
                f"{metrics['recall']:.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR - "
                f"{filepath.name}: "
                f"{e}"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    results = pd.DataFrame(
        all_metrics
    )

    results.to_csv(
        OUTPUT_DIR
        / "ensemble_performance.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("ENSEMBLE RESULTS")
    print("=" * 80)

    if results.empty:

        print(
            "No results generated."
        )

        return

    print(
        f"\nAverage Accuracy: "
        f"{results['accuracy'].mean():.2f}%"
    )

    print(
        f"Average Baseline Accuracy: "
        f"{results['baseline_accuracy'].mean():.2f}%"
    )

    print(
        f"Average Precision: "
        f"{results['precision'].mean():.2f}%"
    )

    print(
        f"Average Recall: "
        f"{results['recall'].mean():.2f}%"
    )

    print(
        f"Average F1: "
        f"{results['f1'].mean():.2f}%"
    )

    print(
        f"Average ROC-AUC: "
        f"{results['roc_auc'].mean():.3f}"
    )

    print("\n")
    print("=" * 80)
    print("STOCK-BY-STOCK RESULTS")
    print("=" * 80)

    print(
        results[
            [
                "symbol",
                "accuracy",
                "baseline_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "actual_positive_rate",
                "predicted_positive_rate",
            ]
        ]
        .sort_values(
            "roc_auc",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("ENSEMBLE BACKTEST COMPLETE")
    print("=" * 80)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()