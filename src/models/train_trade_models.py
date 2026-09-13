from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

HORIZON = 20

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

    df = pd.read_csv(path)

    date_column = None

    for column in [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
    ]:

        if column in df.columns:
            date_column = column
            break

    if date_column is None:
        raise ValueError(
            f"No date column found for {symbol}"
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
# FEATURE SELECTION
# ============================================================

def get_features(df):

    excluded = {
    "date",
    "Future_Return_20D",
    "Buy_Target",
    "Sell_Target",
    "Target",
    "Return_Target",
}

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        # Do not use current live/news overlay
        # for historical model training.
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
# CREATE TARGETS
# ============================================================

def prepare_data(df):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # BUY:
    # Future return greater than +5%
    df["Buy_Target"] = (
        df["Future_Return_20D"]
        > BUY_THRESHOLD
    ).astype(int)

    # SELL:
    # Future return less than -5%
    df["Sell_Target"] = (
        df["Future_Return_20D"]
        < SELL_THRESHOLD
    ).astype(int)

    # Regression target
    df["Return_Target"] = (
        df["Future_Return_20D"]
    )

    return df


# ============================================================
# TRAIN BUY CLASSIFIER
# ============================================================

def train_buy_model(X, y):

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
        X,
        y
    )

    return model


# ============================================================
# TRAIN SELL CLASSIFIER
# ============================================================

def train_sell_model(X, y):

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
        X,
        y
    )

    return model


# ============================================================
# TRAIN RETURN REGRESSOR
# ============================================================

def train_return_model(X, y):

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# TRAIN ONE STOCK
# ============================================================

def train_stock(symbol):

    print("\n" + "=" * 70)
    print(f"TRAINING: {symbol}")
    print("=" * 70)

    df = load_stock(symbol)

    df = prepare_data(df)

    features = get_features(df)

    # Remove rows without complete features
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

    X = df[features]

    buy_y = df["Buy_Target"]

    sell_y = df["Sell_Target"]

    return_y = df["Return_Target"]

    print(
        f"Observations: {len(df)}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"BUY rate: {buy_y.mean() * 100:.2f}%"
    )

    print(
        f"SELL rate: {sell_y.mean() * 100:.2f}%"
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    print("\nTraining BUY model...")

    buy_model = train_buy_model(
        X,
        buy_y
    )

    print("Training SELL model...")

    sell_model = train_sell_model(
        X,
        sell_y
    )

    print("Training return model...")

    return_model = train_return_model(
        X,
        return_y
    )

    # ========================================================
    # SAVE
    # ========================================================

    model_dir = (
        MODELS_DIR
        / f"{symbol.replace('.', '_')}_trade"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        buy_model,
        model_dir / "buy_model.pkl"
    )

    joblib.dump(
        sell_model,
        model_dir / "sell_model.pkl"
    )

    joblib.dump(
        return_model,
        model_dir / "return_model.pkl"
    )

    joblib.dump(
        features,
        model_dir / "features.pkl"
    )

    print(
        f"\nModels saved to:"
        f"\n{model_dir}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature": features,

            "buy_importance":
                buy_model.feature_importances_,

            "sell_importance":
                sell_model.feature_importances_,

            "return_importance":
                return_model.feature_importances_,
        }
    )

    importance["average_importance"] = (
        importance[
            [
                "buy_importance",
                "sell_importance",
                "return_importance",
            ]
        ].mean(axis=1)
    )

    importance.sort_values(
        "average_importance",
        ascending=False,
        inplace=True
    )

    importance.to_csv(
        model_dir
        / "feature_importance.csv",
        index=False
    )

    print("\nTop features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    return {
        "symbol": symbol,
        "observations": len(df),
        "features": len(features),
        "buy_rate": buy_y.mean(),
        "sell_rate": sell_y.mean(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "20-DAY TRADE MODEL TRAINING"
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

    summary = pd.DataFrame(
        results
    )

    summary_path = (
        MODELS_DIR
        / "trade_model_training_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nSuccessfully trained: "
        f"{len(results)}/{len(STOCKS)}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_path}"
    )


if __name__ == "__main__":
    main()