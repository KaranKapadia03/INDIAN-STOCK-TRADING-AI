from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "trade_quality"
)


# ---------------------------------------------------------
# 56 FEATURES
# ---------------------------------------------------------

FEATURES = [
    # Stock / Technical
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

    # Relative Strength
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

    # Future 5-day return
    data[TARGET_RETURN] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    # Trade quality target
    #
    # 1 = stock gains more than +1%
    # 0 = stock gains <= +1%
    #
    data[TARGET] = (
        data[TARGET_RETURN]
        > PROFIT_THRESHOLD
    ).astype(int)

    data.dropna(
        subset=[TARGET_RETURN],
        inplace=True
    )

    return data


def train_stock_model(filepath):

    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    print("\n" + "=" * 70)
    print(f"TRADE QUALITY MODEL: {symbol}")
    print("=" * 70)

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data = create_target(data)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
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

    print(
        f"Positive trades in training: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Positive trades in testing: "
        f"{y_test.mean() * 100:.2f}%"
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    baseline_accuracy = max(
        y_test.mean(),
        1 - y_test.mean()
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions
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
        model,
        model_path / "trade_quality_model.pkl"
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

            "Trade_Probability":
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
        model_path / "predictions.csv"
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    feature_importance.to_csv(
        model_path / "feature_importance.csv",
        index=False
    )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"Accuracy:             "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Baseline Accuracy:    "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Precision:            "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall:               "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score:             "
        f"{f1 * 100:.2f}%"
    )

    if not np.isnan(roc_auc):

        print(
            f"ROC-AUC:              "
            f"{roc_auc:.3f}"
        )

    print("\nConfusion Matrix")

    print(
        matrix
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "NO TRADE",
                "GOOD TRADE"
            ],
            zero_division=0
        )
    )

    print("\nTop 10 Features")

    print("-" * 70)

    print(
        feature_importance
        .head(10)
        .to_string(index=False)
    )

    return {
        "symbol": symbol,
        "accuracy": accuracy,
        "baseline_accuracy":
            baseline_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "positive_rate":
            y_test.mean(),
    }


def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("TRADE QUALITY CLASSIFIER")
    print("=" * 80)

    print(
        f"\nProfit threshold: "
        f"+{PROFIT_THRESHOLD:.1f}%"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    stock_files = sorted(
        FEATURES_DIR.glob("*_features.csv")
    )

    results = []

    successful = 0
    failed = 0

    for filepath in stock_files:

        try:

            result = train_stock_model(
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

        performance_path = (
            PROJECT_ROOT
            / "data"
            / "models"
            / "trade_quality_performance.csv"
        )

        results_df.to_csv(
            performance_path,
            index=False
        )

        print("\n")
        print("=" * 80)
        print("OVERALL RESULTS")
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
            f"Average Positive Rate: "
            f"{results_df['positive_rate'].mean() * 100:.2f}%"
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
                ]
            ].to_string(
                index=False
            )
        )

        print(
            f"\nPerformance saved to:"
            f"\n{performance_path}"
        )

    print("\n")
    print("=" * 80)
    print("TRADE QUALITY TRAINING COMPLETE")
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