from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "regression_relative_strength"


STOCK_FEATURES = [
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
]


MARKET_FEATURES = [
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
]


RELATIVE_STRENGTH_FEATURES = [
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


FEATURES = (
    STOCK_FEATURES
    + MARKET_FEATURES
    + RELATIVE_STRENGTH_FEATURES
)


TARGET = "Future_Return_5D"


def create_target(data):
    data = data.copy()

    data[TARGET] = (
        data["Close"].shift(-5)
        / data["Close"]
        - 1
    ) * 100

    data.dropna(subset=[TARGET], inplace=True)

    return data


def train_stock_model(filepath):
    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    print("\n" + "=" * 70)
    print(f"TRAINING REGRESSION MODEL: {symbol}")
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
            f"Missing features for {symbol}: "
            f"{missing_features}"
        )

    X = data[FEATURES]
    y = data[TARGET]

    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Total rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(FEATURES)}")

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    predictions_df = pd.DataFrame(
        {
            "Actual_Return_5D": y_test.values,
            "Predicted_Return_5D": predictions,
        },
        index=y_test.index
    )

    predictions_df.to_csv(
        model_path / "predictions.csv"
    )

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    feature_importance.to_csv(
        model_path / "feature_importance.csv",
        index=False
    )

    print("\nRESULTS")
    print("-" * 70)
    print(f"MAE:                  {mae:.3f}%")
    print(f"RMSE:                 {rmse:.3f}%")
    print(f"R²:                   {r2:.3f}")
    print(
        f"Directional Accuracy: {directional_accuracy:.2f}%"
    )
    print(f"Correlation:          {correlation:.3f}")
    print(f"Baseline MAE:         {baseline_mae:.3f}%")
    print(f"Baseline RMSE:        {baseline_rmse:.3f}%")
    print(f"Baseline R²:          {baseline_r2:.3f}")

    print("\nTop 10 Features")
    print("-" * 70)
    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    return {
        "symbol": symbol,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
    }


def main():
    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("REGRESSION MODEL — 56 FEATURES")
    print("=" * 80)

    print(f"\nStock features: {len(STOCK_FEATURES)}")
    print(f"Market features: {len(MARKET_FEATURES)}")
    print(
        f"Relative strength features: "
        f"{len(RELATIVE_STRENGTH_FEATURES)}"
    )
    print(f"Total features: {len(FEATURES)}")

    stock_files = sorted(
        FEATURES_DIR.glob("*_features.csv")
    )

    results = []
    successful = 0
    failed = 0

    for filepath in stock_files:

        try:
            result = train_stock_model(filepath)
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

    results_df = pd.DataFrame(results)

    if not results_df.empty:

        performance_path = (
            PROJECT_ROOT
            / "data"
            / "models"
            / "regression_relative_strength_performance.csv"
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
            ].to_string(index=False)
        )

        print(
            f"\nPerformance saved to:"
            f"\n{performance_path}"
        )

    print("\n")
    print("=" * 80)
    print("REGRESSION TRAINING COMPLETE")
    print("=" * 80)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(stock_files)}")


if __name__ == "__main__":
    main()