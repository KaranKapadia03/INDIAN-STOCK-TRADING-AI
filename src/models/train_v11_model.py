from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "v11_logistic_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

TRAIN_RATIO = 0.60

RANDOM_STATE = 42


# ============================================================
# FEATURES WE DO NOT WANT
# ============================================================

EXCLUDE = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    # Forward-looking / target columns
    "Future_Return_20D",
    "Future_Return_5D",
    "Target",
    "Return_Target",
    "Return_20D",

    # Identifiers
    "News_Date",
}


# ============================================================
# SYMBOL
# ============================================================

def get_symbol(path):

    name = path.stem

    name = name.replace(
        "_features_features_regime",
        ""
    )

    name = name.replace(
        "_features_regime",
        ""
    )

    name = name.replace(
        "_features_news",
        ""
    )

    name = name.replace(
        "_features",
        ""
    )

    if name.endswith("_NS"):
        name = (
            name[:-3]
            + ".NS"
        )

    return name


# ============================================================
# LOAD FEATURES
# ============================================================

def get_feature_columns(df):

    columns = []

    for column in df.columns:

        if column in EXCLUDE:
            continue

        if "Future" in str(column):
            continue

        if "Target" in str(column):
            continue

        if str(column).startswith(
            "Unnamed"
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            columns.append(column)

    return columns


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # BUY = positive 20-day return
    df["Target"] = (
        df["Future_Return_20D"] > 0
    ).astype(int)

    df = df.dropna(
        subset=[
            "Future_Return_20D"
        ]
    )

    return df


# ============================================================
# PROCESS
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V11 - FEATURE SELECTION + LOGISTIC REGRESSION")
print("=" * 70)


files = sorted(
    FEATURE_DIR.glob(
        "*_features_regime.csv"
    )
)

if not files:

    files = sorted(
        FEATURE_DIR.glob(
            "*_features_features_regime.csv"
        )
    )


print(
    f"\nFeature files found: {len(files)}"
)


all_predictions = []
all_results = []
all_coefficients = []


# ============================================================
# STOCK LOOP
# ============================================================

for file_path in files:

    symbol = get_symbol(
        file_path
    )

    print("\n" + "-" * 70)
    print(
        f"Processing: {symbol}"
    )

    df = pd.read_csv(
        file_path
    )

    if "Date" not in df.columns:

        print(
            "Skipped: Date column missing"
        )

        continue

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    df = create_target(
        df
    )

    features = get_feature_columns(
        df
    )

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    # Strong market / technical features
    preferred_features = [
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "ATR_14",
        "ATR_Percent",
        "BB_Position",
        "Volume_Ratio",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "SMA20_vs_SMA50",

        "Return_1D",
        "Return_5D",
        "Return_10D",
        "Return_20D",

        "Volatility_20D",

        "NIFTY_SMA_20",
        "NIFTY_SMA_50",
        "NIFTY_SMA_200",
        "NIFTY_Price_vs_SMA50",
        "NIFTY_Price_vs_SMA200",
        "NIFTY_SMA20_vs_SMA50",
        "NIFTY_SMA50_vs_SMA200",
        "NIFTY_Return_20D",
        "NIFTY_Volatility_20D",

        "BANKNIFTY_SMA_20",
        "BANKNIFTY_SMA_50",
        "BANKNIFTY_Price_vs_SMA50",
        "BANKNIFTY_Return_20D",
        "BANKNIFTY_Volatility_20D",

        "Trend_Strength_vs_NIFTY",
        "Trend_Strength_vs_BANKNIFTY",
        "Market_Regime_Score",
    ]

    selected_features = [
        feature
        for feature in preferred_features
        if feature in features
    ]

    # Fallback if some dataset has fewer market features
    if len(selected_features) < 10:

        selected_features = features

    print(
        f"Original features: {len(features)}"
    )

    print(
        f"Selected features: {len(selected_features)}"
    )

    # ========================================================
    # DATA
    # ========================================================

    X = df[
        selected_features
    ].copy()

    y = df["Target"].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns that are completely empty
    usable_columns = [
        column
        for column in X.columns
        if not X[column].isna().all()
    ]

    X = X[
        usable_columns
    ]

    selected_features = usable_columns

    # ========================================================
    # TIME SPLIT
    # ========================================================

    split = int(
        len(X)
        * TRAIN_RATIO
    )

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    # ========================================================
    # MODEL
    # ========================================================

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

    auc = roc_auc_score(
        y_test,
        probabilities
    )

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

    print(
        f"AUC       : {auc:.4f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    all_results.append(
        {
            "Symbol": symbol,
            "Rows": len(df),
            "Original_Features":
                len(features),
            "Selected_Features":
                len(selected_features),
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    test_dates = df[
        "Date"
    ].iloc[
        split:
    ].values

    test_returns = df[
        "Future_Return_20D"
    ].iloc[
        split:
    ].values

    for i in range(
        len(probabilities)
    ):

        all_predictions.append(
            {
                "Date":
                    test_dates[i],

                "Symbol":
                    symbol,

                "Buy_Probability":
                    probabilities[i],

                "Prediction":
                    (
                        "BUY"
                        if predictions[i] == 1
                        else "NO_BUY"
                    ),

                "Actual_20D_Return":
                    test_returns[i],
            }
        )

    # ========================================================
    # FEATURE COEFFICIENTS
    # ========================================================

    logistic_model = (
        model.named_steps[
            "model"
        ]
    )

    coefficients = (
        logistic_model
        .coef_[0]
    )

    for feature, coefficient in zip(
        selected_features,
        coefficients
    ):

        all_coefficients.append(
            {
                "Symbol":
                    symbol,

                "Feature":
                    feature,

                "Coefficient":
                    coefficient,

                "Absolute_Coefficient":
                    abs(coefficient),
            }
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V11 RESULTS")
print("=" * 70)

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

coefficients_df = pd.DataFrame(
    all_coefficients
)


# ============================================================
# SUMMARY
# ============================================================

if not results_df.empty:

    print("\nOverall:")

    print(
        f"Average AUC: "
        f"{results_df['AUC'].mean():.4f}"
    )

    print(
        f"Median AUC: "
        f"{results_df['AUC'].median():.4f}"
    )

    print(
        f"Average Accuracy: "
        f"{results_df['Accuracy'].mean():.4f}"
    )

    print(
        f"Average Precision: "
        f"{results_df['Precision'].mean():.4f}"
    )

    print(
        f"Average Recall: "
        f"{results_df['Recall'].mean():.4f}"
    )

    print(
        f"Average F1: "
        f"{results_df['F1'].mean():.4f}"
    )

    print("\nStock AUC:")

    print(
        results_df[
            [
                "Symbol",
                "AUC",
                "Accuracy"
            ]
        ]
        .sort_values(
            "AUC",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# GLOBAL FEATURE COEFFICIENTS
# ============================================================

if not coefficients_df.empty:

    global_coefficients = (
        coefficients_df
        .groupby("Feature")
        .agg(
            Mean_Coefficient=(
                "Coefficient",
                "mean"
            ),

            Mean_Absolute_Coefficient=(
                "Absolute_Coefficient",
                "mean"
            ),

            Stocks=(
                "Symbol",
                "nunique"
            ),
        )
        .sort_values(
            "Mean_Absolute_Coefficient",
            ascending=False
        )
    )

    print(
        "\nTop features:"
    )

    print(
        global_coefficients
        .head(20)
        .to_string(
            float_format=lambda x:
            f"{x:.5f}"
        )
    )

    global_coefficients.to_csv(
        OUTPUT_DIR
        / "global_coefficients.csv"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR
    / "v11_model_results.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR
    / "v11_predictions.csv",
    index=False
)

coefficients_df.to_csv(
    OUTPUT_DIR
    / "v11_coefficients.csv",
    index=False
)


print("\nFiles saved:")

print(
    OUTPUT_DIR
)

print(
    "  v11_model_results.csv"
)

print(
    "  v11_predictions.csv"
)

print(
    "  v11_coefficients.csv"
)

print(
    "  global_coefficients.csv"
)

print("\n" + "=" * 70)
print("V11 COMPLETE")
print("=" * 70)