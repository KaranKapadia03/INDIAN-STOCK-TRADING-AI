from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "regression_selected"
)


# ---------------------------------------------------------
# SELECTED FEATURES
# ---------------------------------------------------------

SELECTED_FEATURES = [
    # Stock / Technical
    "SMA_50",
    "Volume_SMA_20",
    "ATR_14",
    "Volatility_20D",
    "SMA20_vs_SMA50",
    "EMA_20",
    "MACD",
    "MACD_Histogram",
    "BB_Upper",
    "BB_Lower",

    # NIFTY
    "NIFTY_Volatility_20D",
    "NIFTY_SMA_50",
    "NIFTY_SMA_20",
    "NIFTY_Price_vs_SMA50",

    # BANK NIFTY
    "BANKNIFTY_SMA_50",
    "BANKNIFTY_SMA_20",
    "BANKNIFTY_Volatility_20D",
    "BANKNIFTY_Price_vs_SMA50",

    # Relative Strength
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
    "Relative_Volatility_NIFTY",
    "Relative_Volatility_BANKNIFTY",
]


TARGET = "Future_Return_5D"


def create_target(data):

    data = data.copy()

    data[TARGET] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    data.dropna(
        subset=[TARGET],
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
    print(f"SELECTED FEATURE MODEL: {symbol}")
    print("=" * 70)

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data = create_target(data)

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = data[SELECTED_FEATURES]
    y = data[TARGET]

    # -----------------------------------------------------
    # CHRONOLOGICAL 80/20 SPLIT
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
        f"Selected features: "
        f"{len(SELECTED_FEATURES)}"
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    directional_accuracy = (
        np.sign(predictions)
        == np.sign(y_test.values)
    ).mean() * 100

    correlation = np.corrcoef(
        y_test.values,
        predictions
    )[0, 1]

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    baseline_prediction = np.repeat(
        y_train.mean(),
        len(y_test)
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_prediction
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_prediction
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
        model_path / "return_model.pkl"
    )

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "Actual_Return_5D":
                y_test.values,

            "Predicted_Return_5D":
                predictions,
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
                SELECTED_FEATURES,

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
        f"MAE:                  {mae:.3f}%"
    )

    print(
        f"RMSE:                 {rmse:.3f}%"
    )

    print(
        f"R²:                   {r2:.3f}"
    )

    print(
        f"Directional Accuracy: "
        f"{directional_accuracy:.2f}%"
    )

    print(
        f"Correlation:          "
        f"{correlation:.3f}"
    )

    print(
        f"Baseline MAE:         "
        f"{baseline_mae:.3f}%"
    )

    print(
        f"Baseline RMSE:        "
        f"{baseline_rmse:.3f}%"
    )

    print(
        f"Baseline R²:          "
        f"{baseline_r2:.3f}"
    )

    print("\nTop Features")

    print("-" * 70)

    print(
        feature_importance
        .head(10)
        .to_string(index=False)
    )

    return {
        "symbol": symbol,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy":
            directional_accuracy,
        "correlation": correlation,
        "baseline_mae":
            baseline_mae,
        "baseline_rmse":
            baseline_rmse,
        "baseline_r2":
            baseline_r2,
    }


def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("SELECTED FEATURE REGRESSION")
    print("=" * 80)

    print(
        f"\nUsing "
        f"{len(SELECTED_FEATURES)} "
        f"selected features"
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
    # SAVE OVERALL RESULTS
    # -----------------------------------------------------

    if results:

        results_df = pd.DataFrame(
            results
        )

        performance_path = (
            PROJECT_ROOT
            / "data"
            / "models"
            / "regression_selected_performance.csv"
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
            f"\nAverage MAE: "
            f"{results_df['mae'].mean():.3f}%"
        )

        print(
            f"Average RMSE: "
            f"{results_df['rmse'].mean():.3f}%"
        )

        print(
            f"Average R²: "
            f"{results_df['r2'].mean():.3f}"
        )

        print(
            f"Average Directional Accuracy: "
            f"{results_df['directional_accuracy'].mean():.2f}%"
        )

        print(
            f"Average Correlation: "
            f"{results_df['correlation'].mean():.3f}"
        )

        print(
            f"Average Baseline MAE: "
            f"{results_df['baseline_mae'].mean():.3f}%"
        )

        print(
            f"Average Baseline RMSE: "
            f"{results_df['baseline_rmse'].mean():.3f}%"
        )

        print(
            f"Average Baseline R²: "
            f"{results_df['baseline_r2'].mean():.3f}"
        )

        print("\nPerformance by Stock")
        print("-" * 80)

        print(
            results_df[
                [
                    "symbol",
                    "mae",
                    "rmse",
                    "r2",
                    "directional_accuracy",
                    "correlation",
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
    print("SELECTED FEATURE TRAINING COMPLETE")
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