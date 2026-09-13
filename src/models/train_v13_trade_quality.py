from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
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
    / "v13_trade_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

BUY_THRESHOLD = 5.0
SELL_THRESHOLD = -5.0

TRAIN_RATIO = 0.60

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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
# CREATE TRADE TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # --------------------------------------------------------
    # Three classes
    #
    # BUY  >= +5%
    # HOLD -5% to +5%
    # SELL <= -5%
    # --------------------------------------------------------

    df["Trade_Target"] = "HOLD"

    df.loc[
        df["Future_Return_20D"]
        >= BUY_THRESHOLD,
        "Trade_Target"
    ] = "BUY"

    df.loc[
        df["Future_Return_20D"]
        <= SELL_THRESHOLD,
        "Trade_Target"
    ] = "SELL"

    df = df.dropna(
        subset=[
            "Future_Return_20D"
        ]
    )

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("INDIAN STOCK TRADING AI")
print("V13 - 3-CLASS TRADE QUALITY MODEL")
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

    # --------------------------------------------------------
    # Keep only features that actually exist
    # --------------------------------------------------------

    selected_features = [
        feature
        for feature in FEATURES
        if feature in df.columns
    ]

    X = df[
        selected_features
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df[
        "Trade_Target"
    ]

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    split = int(
        len(df)
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
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    probabilities = model.predict_proba(
        X_test
    )

    predicted_classes = (
        model.predict(
            X_test
        )
    )

    classes = list(
        model.classes_
    )

    # --------------------------------------------------------
    # Get probability for each class
    # --------------------------------------------------------

    probability_map = {
        class_name: probabilities[:, i]
        for i, class_name
        in enumerate(classes)
    }

    buy_probability = (
        probability_map.get(
            "BUY",
            np.zeros(
                len(X_test)
            )
        )
    )

    hold_probability = (
        probability_map.get(
            "HOLD",
            np.zeros(
                len(X_test)
            )
        )
    )

    sell_probability = (
        probability_map.get(
            "SELL",
            np.zeros(
                len(X_test)
            )
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predicted_classes
    )

    f1 = f1_score(
        y_test,
        predicted_classes,
        average="macro"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {f1:.4f}"
    )

    print(
        "\nClassification:"
    )

    print(
        classification_report(
            y_test,
            predicted_classes,
            zero_division=0
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    all_results.append(
        {
            "Symbol":
                symbol,

            "Rows":
                len(df),

            "Features":
                len(selected_features),

            "Accuracy":
                accuracy,

            "Macro_F1":
                f1,

            "BUY_Actual_Percentage":
                (
                    y_test == "BUY"
                ).mean() * 100,

            "HOLD_Actual_Percentage":
                (
                    y_test == "HOLD"
                ).mean() * 100,

            "SELL_Actual_Percentage":
                (
                    y_test == "SELL"
                ).mean() * 100,
        }
    )

    # ========================================================
    # PREDICTION OUTPUT
    # ========================================================

    test_dates = df[
        "Date"
    ].iloc[
        split:
    ].values

    actual_returns = df[
        "Future_Return_20D"
    ].iloc[
        split:
    ].values

    actual_targets = y_test.values

    for i in range(
        len(predicted_classes)
    ):

        all_predictions.append(
            {
                "Date":
                    test_dates[i],

                "Symbol":
                    symbol,

                "BUY_Probability":
                    buy_probability[i],

                "HOLD_Probability":
                    hold_probability[i],

                "SELL_Probability":
                    sell_probability[i],

                "Prediction":
                    predicted_classes[i],

                "Actual_Target":
                    actual_targets[i],

                "Actual_20D_Return":
                    actual_returns[i],
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V13 OVERALL RESULTS")
print("=" * 70)


if not results_df.empty:

    print(
        f"\nAverage Accuracy: "
        f"{results_df['Accuracy'].mean():.4f}"
    )

    print(
        f"Average Macro F1: "
        f"{results_df['Macro_F1'].mean():.4f}"
    )

    print(
        "\nStock results:"
    )

    print(
        results_df[
            [
                "Symbol",
                "Accuracy",
                "Macro_F1",
                "BUY_Actual_Percentage",
                "HOLD_Actual_Percentage",
                "SELL_Actual_Percentage",
            ]
        ]
        .sort_values(
            "Macro_F1",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

if not predictions_df.empty:

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        predictions_df[
            "Prediction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nActual distribution:"
    )

    print(
        predictions_df[
            "Actual_Target"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# BUY SIGNAL ANALYSIS
# ============================================================

if not predictions_df.empty:

    print("\n" + "=" * 70)
    print("BUY SIGNAL QUALITY")
    print("=" * 70)

    buy_signals = predictions_df[
        predictions_df[
            "Prediction"
        ] == "BUY"
    ]

    if len(buy_signals) > 0:

        returns = (
            buy_signals[
                "Actual_20D_Return"
            ]
        )

        print(
            f"\nBUY signals: "
            f"{len(buy_signals)}"
        )

        print(
            f"Average return: "
            f"{returns.mean():.2f}%"
        )

        print(
            f"Median return: "
            f"{returns.median():.2f}%"
        )

        print(
            f"Win rate: "
            f"{(returns > 0).mean() * 100:.2f}%"
        )

        print(
            f">= +5%: "
            f"{(returns >= 5).mean() * 100:.2f}%"
        )

        print(
            f"<= -5%: "
            f"{(returns <= -5).mean() * 100:.2f}%"
        )

    else:

        print(
            "\nNo BUY predictions."
        )


# ============================================================
# SELL SIGNAL ANALYSIS
# ============================================================

if not predictions_df.empty:

    print("\n" + "=" * 70)
    print("SELL SIGNAL QUALITY")
    print("=" * 70)

    sell_signals = predictions_df[
        predictions_df[
            "Prediction"
        ] == "SELL"
    ]

    if len(sell_signals) > 0:

        returns = (
            sell_signals[
                "Actual_20D_Return"
            ]
        )

        print(
            f"\nSELL signals: "
            f"{len(sell_signals)}"
        )

        print(
            f"Average return: "
            f"{returns.mean():.2f}%"
        )

        print(
            f"Median return: "
            f"{returns.median():.2f}%"
        )

        print(
            f"Correct SELL rate: "
            f"{(returns <= -5).mean() * 100:.2f}%"
        )

    else:

        print(
            "\nNo SELL predictions."
        )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR
    / "v13_model_results.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR
    / "v13_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("V13 COMPLETE")
print("=" * 70)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFiles:"
)

print(
    "  v13_model_results.csv"
)

print(
    "  v13_predictions.csv"
)

print("=" * 70)