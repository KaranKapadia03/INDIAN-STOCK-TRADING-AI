import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

FEATURES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)

QUALITY_FILE = os.path.join(
    MODEL_DIR,
    "model_quality.csv"
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "current_trade_predictions.csv"
)


# ============================================================
# TARGET / NON-FEATURE COLUMNS
# ============================================================

TARGET_COLUMNS = {
    "date",
    "Future_Return_20D",
    "Buy_Target",
    "Sell_Target",
    "Target",
    "Return_Target",
}


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(symbol):

    file_path = os.path.join(
        FEATURES_DIR,
        f"{symbol.replace('.', '_')}_features_news.csv"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Feature file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    date_column = None

    for column in df.columns:

        if column.lower() == "date":
            date_column = column
            break

    if date_column is None:

        raise ValueError(
            f"No date column found for {symbol}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df = df.sort_values(
        date_column
    ).reset_index(
        drop=True
    )

    return df, date_column


# ============================================================
# LOAD MODEL QUALITY
# ============================================================

def load_model_quality():

    if not os.path.exists(QUALITY_FILE):

        raise FileNotFoundError(
            f"Model quality file not found:\n"
            f"{QUALITY_FILE}\n\n"
            f"Run:\n"
            f"python src\\models\\model_quality.py"
        )

    quality = pd.read_csv(
        QUALITY_FILE
    )

    quality = quality.set_index(
        "symbol"
    )

    return quality


# ============================================================
# LOAD TRADE MODELS
# ============================================================

def load_trade_models(symbol):

    model_folder = os.path.join(
        MODEL_DIR,
        f"{symbol.replace('.', '_')}_trade"
    )

    buy_model_path = os.path.join(
        model_folder,
        "buy_model.pkl"
    )

    sell_model_path = os.path.join(
        model_folder,
        "sell_model.pkl"
    )

    return_model_path = os.path.join(
        model_folder,
        "return_model.pkl"
    )

    if not os.path.exists(buy_model_path):
        raise FileNotFoundError(
            f"BUY model not found:\n{buy_model_path}"
        )

    if not os.path.exists(sell_model_path):
        raise FileNotFoundError(
            f"SELL model not found:\n{sell_model_path}"
        )

    if not os.path.exists(return_model_path):
        raise FileNotFoundError(
            f"Return model not found:\n{return_model_path}"
        )

    buy_model = joblib.load(
        buy_model_path
    )

    sell_model = joblib.load(
        sell_model_path
    )

    return_model = joblib.load(
        return_model_path
    )

    return (
        buy_model,
        sell_model,
        return_model
    )


# ============================================================
# GET EXACT TRAINING FEATURES
# ============================================================

def get_training_features(model):

    """
    Retrieve the exact feature names used when the
    scikit-learn model was trained.
    """

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "Model does not contain feature_names_in_. "
            "Cannot safely determine training features."
        )

    return list(
        model.feature_names_in_
    )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    latest,
    model
):

    training_features = get_training_features(
        model
    )

    missing_features = [
        feature
        for feature in training_features
        if feature not in latest.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features required by model:\n"
            + "\n".join(
                missing_features
            )
        )

    X = latest[
        training_features
    ].copy()

    return X


# ============================================================
# PREDICT ONE STOCK
# ============================================================

def predict_stock(
    symbol,
    quality
):

    print(
        f"\nProcessing {symbol}..."
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    df, date_column = load_features(
        symbol
    )

    # --------------------------------------------------------
    # Latest row
    # --------------------------------------------------------

    latest = df.iloc[
        [-1]
    ].copy()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        buy_model,
        sell_model,
        return_model
    ) = load_trade_models(
        symbol
    )

    # --------------------------------------------------------
    # Build EXACT model inputs
    #
    # Each model receives only the features it saw
    # during training.
    # --------------------------------------------------------

    X_buy = build_model_input(
        latest,
        buy_model
    )

    X_sell = build_model_input(
        latest,
        sell_model
    )

    X_return = build_model_input(
        latest,
        return_model
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    buy_probability = buy_model.predict_proba(
        X_buy
    )[0][1]

    sell_probability = sell_model.predict_proba(
        X_sell
    )[0][1]

    expected_return = return_model.predict(
        X_return
    )[0]

    # --------------------------------------------------------
    # Model quality
    # --------------------------------------------------------

    buy_auc = quality.loc[
        symbol,
        "buy_auc"
    ]

    sell_auc = quality.loc[
        symbol,
        "sell_auc"
    ]

    buy_quality = quality.loc[
        symbol,
        "buy_quality"
    ]

    sell_quality = quality.loc[
        symbol,
        "sell_quality"
    ]

    overall_quality = quality.loc[
        symbol,
        "overall_quality"
    ]

    ml_weight = quality.loc[
        symbol,
        "ml_weight"
    ]

    buy_enabled = bool(
        quality.loc[
            symbol,
            "buy_enabled"
        ]
    )

    sell_enabled = bool(
        quality.loc[
            symbol,
            "sell_enabled"
        ]
    )

    # --------------------------------------------------------
    # Quality-adjusted BUY probability
    # --------------------------------------------------------

    if not buy_enabled:

        effective_buy_probability = 0.50

    else:

        effective_buy_probability = (
            0.50
            + (
                buy_probability - 0.50
            ) * ml_weight
        )

    # --------------------------------------------------------
    # Quality-adjusted SELL probability
    # --------------------------------------------------------

    if not sell_enabled:

        effective_sell_probability = 0.50

    else:

        effective_sell_probability = (
            0.50
            + (
                sell_probability - 0.50
            ) * ml_weight
        )

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    price = latest[
        "Close"
    ].iloc[0]

    # --------------------------------------------------------
    # Current date
    # --------------------------------------------------------

    current_date = latest[
        date_column
    ].iloc[0]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "symbol": symbol,

        "date": current_date,

        "price": price,

        # Raw predictions
        "buy_probability": buy_probability,

        "sell_probability": sell_probability,

        "expected_return_20d": expected_return,

        # Validation
        "buy_auc": buy_auc,

        "sell_auc": sell_auc,

        "buy_quality": buy_quality,

        "sell_quality": sell_quality,

        "overall_quality": overall_quality,

        "ml_weight": ml_weight,

        # Adjusted predictions
        "effective_buy_probability":
            effective_buy_probability,

        "effective_sell_probability":
            effective_sell_probability,

        "buy_enabled":
            buy_enabled,

        "sell_enabled":
            sell_enabled,
    }

    print(
        f"{symbol:<15}"
        f" BUY={buy_probability:6.2%}"
        f" SELL={sell_probability:6.2%}"
        f" RETURN={expected_return:+6.2f}%"
        f" QUALITY={overall_quality}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("QUALITY-AWARE TRADE PREDICTIONS")
    print("=" * 100)

    # --------------------------------------------------------
    # Load model quality
    # --------------------------------------------------------

    quality = load_model_quality()

    symbols = quality.index.tolist()

    results = []

    # --------------------------------------------------------
    # Process all stocks
    # --------------------------------------------------------

    for symbol in symbols:

        try:

            result = predict_stock(
                symbol,
                quality
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                f"\nERROR: {symbol}"
            )

            print(
                str(e)
            )

    # --------------------------------------------------------
    # Make sure predictions exist
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No stock predictions were generated."
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by expected return
    # --------------------------------------------------------

    predictions = predictions.sort_values(
        "expected_return_20d",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 130)
    print("CURRENT QUALITY-AWARE PREDICTIONS")
    print("=" * 130)

    display_columns = [

        "symbol",

        "price",

        "buy_probability",

        "sell_probability",

        "effective_buy_probability",

        "effective_sell_probability",

        "expected_return_20d",

        "overall_quality",

        "ml_weight",
    ]

    print(
        predictions[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "Stocks processed:",
        len(predictions)
    )

    print(
        "BUY ML enabled:",
        int(
            predictions[
                "buy_enabled"
            ].sum()
        )
    )

    print(
        "SELL ML enabled:",
        int(
            predictions[
                "sell_enabled"
            ].sum()
        )
    )

    print("\nTop expected-return stocks:")

    print(
        predictions[
            [
                "symbol",
                "expected_return_20d",
                "effective_buy_probability",
                "effective_sell_probability",
                "overall_quality",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILE SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()