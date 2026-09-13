from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/models/v14_regression")

TRAIN_RATIO = 0.60
HORIZON = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Features selected from V10/V11 diagnostics
FEATURES = [
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
    "BB_Position",
    "ATR_Percent",
    "Volume_Ratio",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",

    # Market context
    "NIFTY_Return_20D",
    "NIFTY_SMA_20",
    "NIFTY_SMA_50",
    "NIFTY_SMA_200",
    "NIFTY_Price_vs_SMA50",
    "NIFTY_Price_vs_SMA200",
    "NIFTY_SMA20_vs_SMA50",
    "NIFTY_SMA50_vs_SMA200",

    "BANKNIFTY_Return_20D",
    "BANKNIFTY_SMA_20",
    "BANKNIFTY_SMA_50",

    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
    "Market_Regime_Score",
]


# ============================================================
# HELPERS
# ============================================================

def get_symbol(file_path):
    name = file_path.stem

    name = name.replace("_features_features_regime", "")
    name = name.replace("_features_regime", "")
    name = name.replace("_features", "")

    if name.endswith("_NS"):
        name = name[:-3] + ".NS"

    return name


def create_target(df):
    """
    Predict the percentage return over the next 20 trading days.
    """

    df = df.copy()

    df["Target_Return_20D"] = (
        df["Close"].shift(-HORIZON) / df["Close"] - 1
    ) * 100

    df.dropna(subset=["Target_Return_20D"], inplace=True)

    return df


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    direction_accuracy = (
        np.sign(predictions) == np.sign(y_test)
    ).mean()

    correlation = np.corrcoef(predictions, y_test)[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Direction_Accuracy": direction_accuracy,
        "Correlation": correlation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("V14 - 20D RETURN REGRESSION MODEL")
    print("=" * 70)

    files = sorted(FEATURE_DIR.glob("*_features.csv"))

    # Remove duplicate/regime feature files
    files = [
        f for f in files
        if "_regime" not in f.stem
        and "_news" not in f.stem
    ]

    print(f"\nFeature files found: {len(files)}")

    all_results = []
    all_predictions = []

    for file_path in files:

        symbol = get_symbol(file_path)

        print("\n" + "-" * 70)
        print(f"Processing: {symbol}")

        df = pd.read_csv(file_path)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        df = create_target(df)

        available_features = [
            f for f in FEATURES
            if f in df.columns
        ]

        if len(available_features) < 10:
            print("Not enough features. Skipping.")
            continue

        data = df[
            ["Date", "Close", "Target_Return_20D"]
            + available_features
        ].copy()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if len(data) < 500:
            print("Not enough rows. Skipping.")
            continue

        split = int(len(data) * TRAIN_RATIO)

        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()

        X_train = train[available_features]
        y_train = train["Target_Return_20D"]

        X_test = test[available_features]
        y_test = test["Target_Return_20D"]

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        models = {

            "Ridge": Ridge(
                alpha=10.0
            ),

            "RandomForest": RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ),

            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=8,
                random_state=42
            ),
        }

        stock_results = []

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            metrics = evaluate(
                model,
                X_test,
                y_test
            )

            print(
                f"{model_name:18s} "
                f"MAE={metrics['MAE']:.3f}% | "
                f"RMSE={metrics['RMSE']:.3f}% | "
                f"R²={metrics['R2']:.3f} | "
                f"Direction={metrics['Direction_Accuracy']:.3f} | "
                f"Corr={metrics['Correlation']:.3f}"
            )

            result = {
                "Symbol": symbol,
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Direction_Accuracy": metrics["Direction_Accuracy"],
                "Correlation": metrics["Correlation"],
            }

            all_results.append(result)
            stock_results.append(result)

            # Save predictions
            predictions = model.predict(X_test)

            pred_df = pd.DataFrame({
                "Date": test["Date"].values,
                "Symbol": symbol,
                "Model": model_name,
                "Actual_Return_20D": y_test.values,
                "Predicted_Return_20D": predictions,
            })

            all_predictions.append(pred_df)

        # ----------------------------------------------------
        # SELECT BEST MODEL FOR THIS STOCK
        # ----------------------------------------------------

        best = max(
            stock_results,
            key=lambda x: x["Correlation"]
        )

        print(
            f"BEST MODEL: {best['Model']} "
            f"(Correlation={best['Correlation']:.3f})"
        )

        # Retrain best model on all available data
        best_name = best["Model"]
        best_model = models[best_name]

        X_all = data[available_features]
        y_all = data["Target_Return_20D"]

        best_model.fit(X_all, y_all)

        model_path = (
            OUTPUT_DIR /
            f"{symbol.replace('.', '_')}_best_model.joblib"
        )

        joblib.dump(
            {
                "model": best_model,
                "features": available_features,
                "symbol": symbol,
                "horizon": HORIZON,
            },
            model_path
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    results_df.to_csv(
        OUTPUT_DIR / "v14_model_results.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "v14_predictions.csv",
        index=False
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("V14 OVERALL RESULTS")
    print("=" * 70)

    summary = (
        results_df
        .groupby("Model")
        [
            [
                "MAE",
                "RMSE",
                "R2",
                "Direction_Accuracy",
                "Correlation",
            ]
        ]
        .mean()
        .sort_values(
            "Correlation",
            ascending=False
        )
    )

    print("\nAverage Model Performance:\n")
    print(summary.round(4))

    print("\n")
    print("=" * 70)
    print("V14 COMPLETE")
    print("=" * 70)

    print("\nSaved to:")
    print(OUTPUT_DIR.resolve())

    print("\nFiles:")
    print("  v14_model_results.csv")
    print("  v14_predictions.csv")
    print("  *_best_model.joblib")


if __name__ == "__main__":
    main()