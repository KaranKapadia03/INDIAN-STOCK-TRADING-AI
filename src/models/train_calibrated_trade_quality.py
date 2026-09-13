from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "calibrated_trade_quality"
)

PERFORMANCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "calibrated_trade_quality_performance.csv"
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

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

    "NIFTY_Close",
    "NIFTY_Return_1D",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",
    "NIFTY_SMA_20",
    "NIFTY_SMA_50",
    "NIFTY_Volatility_20D",
    "NIFTY_Price_vs_SMA20",
    "NIFTY_Price_vs_SMA50",

    "BANKNIFTY_Close",
    "BANKNIFTY_Return_1D",
    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",
    "BANKNIFTY_SMA_20",
    "BANKNIFTY_SMA_50",
    "BANKNIFTY_Volatility_20D",
    "BANKNIFTY_Price_vs_SMA20",
    "BANKNIFTY_Price_vs_SMA50",

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


TARGET_RETURN = "Future_Return_5D"

TARGET = "Trade_Quality"

PROFIT_THRESHOLD = 1.0


def create_target(data):

    data = data.copy()

    data[TARGET_RETURN] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    data[TARGET] = (
        data[TARGET_RETURN]
        > PROFIT_THRESHOLD
    ).astype(int)

    data.dropna(
        subset=[TARGET_RETURN],
        inplace=True
    )

    return data


def train_stock(filepath):

    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    print("\n" + "=" * 70)
    print(f"CALIBRATED MODEL: {symbol}")
    print("=" * 70)

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data = create_target(data)

    missing = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = data[FEATURES]
    y = data[TARGET]

    # -----------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # -----------------------------------------------------

    split_index = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Total rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # -----------------------------------------------------
    # BASE RANDOM FOREST
    # -----------------------------------------------------

    base_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # -----------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------
    #
    # Important:
    # cv=5 performs cross-validation ONLY inside
    # the training dataset.
    #
    # The final 20% remains completely untouched
    # for evaluation.
    #
    # sigmoid = Platt scaling
    #

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    predictions = calibrated_model.predict(
        X_test
    )

    probabilities = (
        calibrated_model
        .predict_proba(X_test)[:, 1]
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    if y_test.nunique() > 1:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        roc_auc = np.nan

    brier_score = brier_score_loss(
        y_test,
        probabilities
    )

    baseline_accuracy = max(
        y_test.mean(),
        1 - y_test.mean()
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / symbol.replace(".", "_")
    )

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        calibrated_model,
        model_path
        / "calibrated_trade_quality_model.pkl"
    )

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "Actual_Trade_Quality":
                y_test.values,

            "Predicted_Trade_Quality":
                predictions,

            "Calibrated_Probability":
                probabilities,

            "Actual_Future_Return_5D":
                data.loc[
                    y_test.index,
                    TARGET_RETURN
                ].values,
        },
        index=y_test.index
    )

    predictions_df.to_csv(
        model_path
        / "predictions.csv"
    )

    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"Accuracy:          "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Precision:         "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall:            "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score:          "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:           "
        f"{roc_auc:.3f}"
    )

    print(
        f"Brier Score:       "
        f"{brier_score:.4f}"
    )

    print(
        f"Average Probability: "
        f"{probabilities.mean() * 100:.2f}%"
    )

    return {
        "symbol": symbol,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "average_probability":
            probabilities.mean(),
    }


def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("CALIBRATED TRADE QUALITY MODELS")
    print("=" * 80)

    print(
        f"\nProfit threshold: "
        f"+{PROFIT_THRESHOLD:.1f}%"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    stock_files = sorted(
        FEATURES_DIR.glob(
            "*_features.csv"
        )
    )

    results = []

    successful = 0
    failed = 0

    for filepath in stock_files:

        try:

            result = train_stock(
                filepath
            )

            results.append(result)

            successful += 1

        except Exception as e:

            symbol = (
                filepath.stem
                .replace("_features", "")
                .replace("_", ".")
            )

            print(
                f"\nERROR - {symbol}: {e}"
            )

            failed += 1

    # -----------------------------------------------------
    # OVERALL RESULTS
    # -----------------------------------------------------

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            PERFORMANCE_FILE,
            index=False
        )

        print("\n")
        print("=" * 80)
        print("OVERALL CALIBRATED MODEL RESULTS")
        print("=" * 80)

        print(
            f"\nAverage Accuracy: "
            f"{results_df['accuracy'].mean() * 100:.2f}%"
        )

        print(
            f"Average Baseline Accuracy: "
            f"{results_df['baseline_accuracy'].mean() * 100:.2f}%"
        )

        print(
            f"Average Precision: "
            f"{results_df['precision'].mean() * 100:.2f}%"
        )

        print(
            f"Average Recall: "
            f"{results_df['recall'].mean() * 100:.2f}%"
        )

        print(
            f"Average F1: "
            f"{results_df['f1'].mean() * 100:.2f}%"
        )

        print(
            f"Average ROC-AUC: "
            f"{results_df['roc_auc'].mean():.3f}"
        )

        print(
            f"Average Brier Score: "
            f"{results_df['brier_score'].mean():.4f}"
        )

        print("\nPerformance by Stock")

        print("-" * 80)

        print(
            results_df[
                [
                    "symbol",
                    "accuracy",
                    "baseline_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "brier_score",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            f"\nSaved to:"
            f"\n{PERFORMANCE_FILE}"
        )

    print("\n")
    print("=" * 80)
    print("CALIBRATED TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total: {len(stock_files)}"
    )


if __name__ == "__main__":
    main()