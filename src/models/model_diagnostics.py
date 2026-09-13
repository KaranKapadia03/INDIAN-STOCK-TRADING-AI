from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "v10_model_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 20

TRAIN_RATIO = 0.60

RANDOM_STATE = 42


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

EXCLUDE_COLUMNS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    # Future information
    "Future_Return_20D",
    "Future_Return_5D",
    "Target",

    # Anything explicitly target-related
    "Return_Target",
    "Return_20D",

    # News aggregation identifiers
    "News_Date",
}


# ============================================================
# HELPERS
# ============================================================

def symbol_from_file(path):

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


def build_target(df):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # Binary direction target.
    df["Target"] = (
        df["Future_Return_20D"] > 0
    ).astype(int)

    return df


def get_features(df):

    features = []

    for column in df.columns:

        if column in EXCLUDE_COLUMNS:
            continue

        if column == "Target":
            continue

        if "Future" in column:
            continue

        if "Target" in column:
            continue

        if str(column).startswith("Unnamed"):
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(column)

    return features


def clean_features(df, features):

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.ffill()

    X = X.fillna(
        X.median()
    )

    return X


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

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

    result = {
        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "AUC":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "Probability_Mean":
            probabilities.mean(),

        "Probability_Std":
            probabilities.std(),
    }

    return result, model


# ============================================================
# START
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V10 - MODEL DIAGNOSTICS & FEATURE SELECTION")
print("=" * 70)


# ============================================================
# FIND FEATURE FILES
# ============================================================

files = sorted(
    FEATURE_DIR.glob(
        "*_features_regime.csv"
    )
)

if not files:

    # Fallback for the actual filenames
    files = sorted(
        FEATURE_DIR.glob(
            "*_features_features_regime.csv"
        )
    )

print(
    f"\nFeature files found: "
    f"{len(files)}"
)


# ============================================================
# RESULTS
# ============================================================

all_model_results = []
all_importance = []
all_correlations = []


# ============================================================
# PROCESS EACH STOCK
# ============================================================

for file_path in files:

    symbol = symbol_from_file(
        file_path
    )

    print("\n" + "-" * 70)

    print(
        f"Processing: {symbol}"
    )

    df = pd.read_csv(
        file_path
    )

    date_col = None

    for column in df.columns:

        if str(column).lower() == "date":

            date_col = column
            break

    if date_col is None:

        print(
            "Skipping - no Date column."
        )

        continue

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df = df.sort_values(
        date_col
    )

    df = build_target(df)

    df = df.dropna(
        subset=[
            "Future_Return_20D"
        ]
    )

    features = get_features(
        df
    )

    if len(features) < 5:

        print(
            "Skipping - insufficient features."
        )

        continue

    X = clean_features(
        df,
        features
    )

    y = df["Target"]

    # --------------------------------------------------------
    # Remove rows where target is invalid
    # --------------------------------------------------------

    valid = (
        y.notna()
        &
        X.notna().all(axis=1)
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 300:

        print(
            "Skipping - insufficient observations."
        )

        continue

    # --------------------------------------------------------
    # Time-series split
    # --------------------------------------------------------

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

    print(
        f"Rows: {len(X):,}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Train: {len(X_train):,}"
    )

    print(
        f"Test: {len(X_test):,}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "RandomForest":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),

        "LogisticRegression":
            Pipeline(
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
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
    }

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for model_name, model in models.items():

        try:

            result, fitted_model = evaluate_model(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            result["Symbol"] = symbol
            result["Features"] = len(features)
            result["Train_Rows"] = len(X_train)
            result["Test_Rows"] = len(X_test)

            all_model_results.append(
                result
            )

            print(
                f"{model_name:<20}"
                f"AUC={result['AUC']:.4f} "
                f"Accuracy={result['Accuracy']:.4f}"
            )

            # ------------------------------------------------
            # Feature importance
            # ------------------------------------------------

            if model_name == "RandomForest":

                importance = (
                    fitted_model
                    .feature_importances_
                )

                for feature, value in zip(
                    features,
                    importance
                ):

                    all_importance.append(
                        {
                            "Symbol":
                                symbol,
                            "Feature":
                                feature,
                            "Importance":
                                value,
                        }
                    )

        except Exception as exc:

            print(
                f"{model_name} failed: "
                f"{exc}"
            )

    # ========================================================
    # FEATURE / RETURN CORRELATION
    # ========================================================

    correlation_df = X.copy()

    correlation_df[
        "Future_Return_20D"
    ] = df.loc[
        correlation_df.index,
        "Future_Return_20D"
    ]

    correlations = (
        correlation_df
        .corr()["Future_Return_20D"]
        .drop(
            "Future_Return_20D"
        )
        .sort_values(
            key=lambda x:
            x.abs(),
            ascending=False
        )
    )

    for feature, value in correlations.items():

        all_correlations.append(
            {
                "Symbol":
                    symbol,
                "Feature":
                    feature,
                "Correlation":
                    value,
                "Abs_Correlation":
                    abs(value),
            }
        )


# ============================================================
# MODEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    all_model_results
)

if not results_df.empty:

    summary = (
        results_df
        .groupby("Model")
        .agg(
            Stocks=("Symbol", "count"),
            Average_AUC=("AUC", "mean"),
            Median_AUC=("AUC", "median"),
            Average_Accuracy=(
                "Accuracy",
                "mean"
            ),
            Average_Precision=(
                "Precision",
                "mean"
            ),
            Average_Recall=(
                "Recall",
                "mean"
            ),
            Average_F1=(
                "F1",
                "mean"
            ),
        )
        .sort_values(
            "Average_AUC",
            ascending=False
        )
    )

    print(
        summary.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "model_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR
        / "model_summary.csv"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame(
    all_importance
)

if not importance_df.empty:

    global_importance = (
        importance_df
        .groupby("Feature")
        .agg(
            Mean_Importance=(
                "Importance",
                "mean"
            ),
            Median_Importance=(
                "Importance",
                "median"
            ),
            Stocks=(
                "Symbol",
                "nunique"
            ),
        )
        .sort_values(
            "Mean_Importance",
            ascending=False
        )
    )

    print(
        global_importance
        .head(25)
        .to_string(
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    importance_df.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_stock.csv",
        index=False
    )

    global_importance.to_csv(
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )


# ============================================================
# GLOBAL CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL FEATURE / FUTURE RETURN CORRELATION")
print("=" * 70)

correlation_df = pd.DataFrame(
    all_correlations
)

if not correlation_df.empty:

    global_corr = (
        correlation_df
        .groupby("Feature")
        .agg(
            Mean_Correlation=(
                "Correlation",
                "mean"
            ),
            Mean_Absolute_Correlation=(
                "Abs_Correlation",
                "mean"
            ),
            Stocks=(
                "Symbol",
                "nunique"
            ),
        )
        .sort_values(
            "Mean_Absolute_Correlation",
            ascending=False
        )
    )

    print(
        global_corr
        .head(25)
        .to_string(
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    correlation_df.to_csv(
        OUTPUT_DIR
        / "feature_correlations_by_stock.csv",
        index=False
    )

    global_corr.to_csv(
        OUTPUT_DIR
        / "global_feature_correlations.csv"
    )


# ============================================================
# BEST MODEL
# ============================================================

if not results_df.empty:

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    best_model = (
        summary
        .sort_values(
            "Average_AUC",
            ascending=False
        )
        .iloc[0]
    )

    print(
        f"\nModel: "
        f"{best_model.name}"
    )

    print(
        f"Average AUC: "
        f"{best_model['Average_AUC']:.4f}"
    )

    print(
        f"Median AUC: "
        f"{best_model['Median_AUC']:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V10 MODEL DIAGNOSTICS COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFiles created:")

print("  model_results.csv")
print("  model_summary.csv")
print("  feature_importance_by_stock.csv")
print("  global_feature_importance.csv")
print("  feature_correlations_by_stock.csv")
print("  global_feature_correlations.csv")

print("\n" + "=" * 70)