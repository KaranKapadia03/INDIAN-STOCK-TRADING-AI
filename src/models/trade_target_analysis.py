from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "trade_target_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

INITIAL_TRAIN_RATIO = 0.60

RETRAIN_DAYS = 60

TEST_WINDOW_DAYS = 60

BUY_THRESHOLD = 5.0

SELL_THRESHOLD = -5.0


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
        "Buy_Target",
        "Sell_Target",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        # Historical news is not reliable in our dataset.
        # Therefore do not use it for this backtest.
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
# PREPARE TARGETS
# ============================================================

def prepare_data(df, features):

    df = df.copy()

    # --------------------------------------------------------
    # Future 20-day return
    # --------------------------------------------------------

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # --------------------------------------------------------
    # BUY target
    #
    # 1 = future return > +5%
    # 0 = everything else
    # --------------------------------------------------------

    df["Buy_Target"] = (
        df["Future_Return_20D"]
        > BUY_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # SELL target
    #
    # 1 = future return < -5%
    # 0 = everything else
    # --------------------------------------------------------

    df["Sell_Target"] = (
        df["Future_Return_20D"]
        < SELL_THRESHOLD
    ).astype(int)

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
# TRAIN ONE BINARY MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_test,
):

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
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return (
        model,
        predictions,
        probabilities,
    )


# ============================================================
# EVALUATE BINARY MODEL
# ============================================================

def evaluate_model(
    actual,
    predictions,
    probabilities,
):

    accuracy = accuracy_score(
        actual,
        predictions
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    positive_rate = actual.mean()

    baseline = max(
        positive_rate,
        1 - positive_rate
    )

    try:

        auc = roc_auc_score(
            actual,
            probabilities
        )

    except ValueError:

        auc = np.nan

    return {
        "accuracy": accuracy,
        "baseline_accuracy": baseline,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "positive_rate": positive_rate,
    }


# ============================================================
# WALK-FORWARD ANALYSIS
# ============================================================

def analyze_stock(symbol):

    print("\n" + "=" * 75)
    print(
        f"TRADE TARGET ANALYSIS: {symbol}"
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

    buy_predictions = []

    sell_predictions = []

    train_end = initial_train_size

    fold = 0

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

        if test.empty:

            break

        X_train = train[
            features
        ]

        X_test = test[
            features
        ]

        # ====================================================
        # BUY MODEL
        # ====================================================

        buy_model, buy_pred, buy_prob = (
            train_model(
                X_train,
                train["Buy_Target"],
                X_test
            )
        )

        buy_result = pd.DataFrame(
            {
                "date":
                    test["date"].values,

                "symbol":
                    symbol,

                "Future_Return_20D":
                    test[
                        "Future_Return_20D"
                    ].values,

                "Buy_Actual":
                    test[
                        "Buy_Target"
                    ].values,

                "Buy_Prediction":
                    buy_pred,

                "Buy_Probability":
                    buy_prob,

                "Fold":
                    fold,
            }
        )

        buy_predictions.append(
            buy_result
        )

        # ====================================================
        # SELL MODEL
        # ====================================================

        sell_model, sell_pred, sell_prob = (
            train_model(
                X_train,
                train["Sell_Target"],
                X_test
            )
        )

        sell_result = pd.DataFrame(
            {
                "date":
                    test["date"].values,

                "symbol":
                    symbol,

                "Future_Return_20D":
                    test[
                        "Future_Return_20D"
                    ].values,

                "Sell_Actual":
                    test[
                        "Sell_Target"
                    ].values,

                "Sell_Prediction":
                    sell_pred,

                "Sell_Probability":
                    sell_prob,

                "Fold":
                    fold,
            }
        )

        sell_predictions.append(
            sell_result
        )

        print(
            f"Fold {fold + 1}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"{test['date'].iloc[0].date()} "
            f"to "
            f"{test['date'].iloc[-1].date()}"
        )

        fold += 1

        train_end += RETRAIN_DAYS

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    buy_df = pd.concat(
        buy_predictions,
        ignore_index=True
    )

    sell_df = pd.concat(
        sell_predictions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # BUY metrics
    # --------------------------------------------------------

    buy_metrics = evaluate_model(
        buy_df["Buy_Actual"],
        buy_df["Buy_Prediction"],
        buy_df["Buy_Probability"],
    )

    # --------------------------------------------------------
    # SELL metrics
    # --------------------------------------------------------

    sell_metrics = evaluate_model(
        sell_df["Sell_Actual"],
        sell_df["Sell_Prediction"],
        sell_df["Sell_Probability"],
    )

    # ========================================================
    # THRESHOLD PERFORMANCE
    # ========================================================

    buy_threshold_results = []

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        subset = buy_df[
            buy_df[
                "Buy_Probability"
            ] >= threshold
        ]

        if subset.empty:

            continue

        buy_threshold_results.append(
            {
                "symbol":
                    symbol,

                "type":
                    "BUY",

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

                "meaningful_buy_rate_pct":
                    (
                        subset[
                            "Future_Return_20D"
                        ]
                        > BUY_THRESHOLD
                    )
                    .mean()
                    * 100,
            }
        )

    sell_threshold_results = []

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        subset = sell_df[
            sell_df[
                "Sell_Probability"
            ] >= threshold
        ]

        if subset.empty:

            continue

        sell_threshold_results.append(
            {
                "symbol":
                    symbol,

                "type":
                    "SELL",

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
                        .lt(0)
                        .mean()
                        * 100
                    ),

                "meaningful_sell_rate_pct":
                    (
                        subset[
                            "Future_Return_20D"
                        ]
                        < SELL_THRESHOLD
                    )
                    .mean()
                    * 100,
            }
        )

    threshold_df = pd.DataFrame(
        buy_threshold_results
        + sell_threshold_results
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    buy_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_buy_predictions.csv"
    )

    sell_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_sell_predictions.csv"
    )

    threshold_file = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_thresholds.csv"
    )

    buy_df.to_csv(
        buy_file,
        index=False
    )

    sell_df.to_csv(
        sell_file,
        index=False
    )

    threshold_df.to_csv(
        threshold_file,
        index=False
    )

    print("\nBUY MODEL")

    print(
        f"Positive rate: "
        f"{buy_metrics['positive_rate']:.3f}"
    )

    print(
        f"Accuracy: "
        f"{buy_metrics['accuracy']:.4f}"
    )

    print(
        f"Baseline: "
        f"{buy_metrics['baseline_accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{buy_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{buy_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{buy_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{buy_metrics['auc']:.4f}"
    )

    print("\nSELL MODEL")

    print(
        f"Positive rate: "
        f"{sell_metrics['positive_rate']:.3f}"
    )

    print(
        f"Accuracy: "
        f"{sell_metrics['accuracy']:.4f}"
    )

    print(
        f"Baseline: "
        f"{sell_metrics['baseline_accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{sell_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{sell_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{sell_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{sell_metrics['auc']:.4f}"
    )

    return {
        "symbol":
            symbol,

        "buy_auc":
            buy_metrics["auc"],

        "buy_accuracy":
            buy_metrics["accuracy"],

        "buy_baseline":
            buy_metrics["baseline_accuracy"],

        "buy_precision":
            buy_metrics["precision"],

        "buy_recall":
            buy_metrics["recall"],

        "buy_f1":
            buy_metrics["f1"],

        "buy_positive_rate":
            buy_metrics["positive_rate"],

        "sell_auc":
            sell_metrics["auc"],

        "sell_accuracy":
            sell_metrics["accuracy"],

        "sell_baseline":
            sell_metrics["baseline_accuracy"],

        "sell_precision":
            sell_metrics["precision"],

        "sell_recall":
            sell_metrics["recall"],

        "sell_f1":
            sell_metrics["f1"],

        "sell_positive_rate":
            sell_metrics["positive_rate"],
    }, threshold_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "BUY / SELL TARGET ANALYSIS"
    )
    print("=" * 75)

    results = []

    all_thresholds = []

    for symbol in STOCKS:

        try:

            result, thresholds = (
                analyze_stock(symbol)
            )

            results.append(
                result
            )

            all_thresholds.append(
                thresholds
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

    thresholds = pd.concat(
        all_thresholds,
        ignore_index=True
    )

    summary_file = (
        OUTPUT_DIR
        / "trade_target_summary.csv"
    )

    threshold_file = (
        OUTPUT_DIR
        / "all_threshold_results.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    thresholds.to_csv(
        threshold_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print(
        "TRADE TARGET SUMMARY"
    )
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Average BUY ROC-AUC: "
        f"{summary['buy_auc'].mean():.4f}"
    )

    print(
        f"Average SELL ROC-AUC: "
        f"{summary['sell_auc'].mean():.4f}"
    )

    print("\n")
    print("=" * 75)
    print(
        "BEST BUY MODELS"
    )
    print("=" * 75)

    print(
        summary.sort_values(
            "buy_auc",
            ascending=False
        )[
            [
                "symbol",
                "buy_auc",
                "buy_accuracy",
                "buy_baseline",
                "buy_precision",
                "buy_recall",
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
        "BEST SELL MODELS"
    )
    print("=" * 75)

    print(
        summary.sort_values(
            "sell_auc",
            ascending=False
        )[
            [
                "symbol",
                "sell_auc",
                "sell_accuracy",
                "sell_baseline",
                "sell_precision",
                "sell_recall",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # HIGH-CONFIDENCE BUY
    # ========================================================

    print("\n")
    print("=" * 75)
    print(
        "BUY PROBABILITY PERFORMANCE"
    )
    print("=" * 75)

    buy_thresholds = thresholds[
        thresholds["type"] == "BUY"
    ]

    buy_grouped = (
        buy_thresholds
        .groupby("threshold")
        .agg(
            observations=(
                "observations",
                "sum"
            ),

            average_return_pct=(
                "average_return_pct",
                "mean"
            ),

            meaningful_buy_rate_pct=(
                "meaningful_buy_rate_pct",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        buy_grouped.to_string(
            index=False
        )
    )

    # ========================================================
    # HIGH-CONFIDENCE SELL
    # ========================================================

    print("\n")
    print("=" * 75)
    print(
        "SELL PROBABILITY PERFORMANCE"
    )
    print("=" * 75)

    sell_thresholds = thresholds[
        thresholds["type"] == "SELL"
    ]

    sell_grouped = (
        sell_thresholds
        .groupby("threshold")
        .agg(
            observations=(
                "observations",
                "sum"
            ),

            average_return_pct=(
                "average_return_pct",
                "mean"
            ),

            meaningful_sell_rate_pct=(
                "meaningful_sell_rate_pct",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        sell_grouped.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 75)
    print(
        "TRADE TARGET ANALYSIS COMPLETE"
    )
    print("=" * 75)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()