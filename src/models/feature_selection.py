from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "feature_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


RELATIVE_FEATURES = [
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
    + RELATIVE_FEATURES
)


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


def analyze_stock(filepath):

    symbol = (
        filepath.stem
        .replace("_features", "")
        .replace("_", ".")
    )

    print("\n" + "=" * 70)
    print(f"FEATURE ANALYSIS: {symbol}")
    print("=" * 70)

    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    data = create_target(data)

    missing = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = data[FEATURES]
    y = data[TARGET]

    split_index = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

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

    # --------------------------------------------------
    # RANDOM FOREST IMPURITY IMPORTANCE
    # --------------------------------------------------

    rf_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "rf_importance":
                model.feature_importances_,
        }
    )

    # --------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------

    print("Calculating permutation importance...")

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    permutation_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "permutation_importance":
                permutation.importances_mean,
        }
    )

    # --------------------------------------------------
    # COMBINE
    # --------------------------------------------------

    importance = rf_importance.merge(
        permutation_df,
        on="feature"
    )

    importance["combined_score"] = (
        importance["rf_importance"]
        * (
            1
            + importance[
                "permutation_importance"
            ].clip(lower=0)
        )
    )

    importance.sort_values(
        "combined_score",
        ascending=False,
        inplace=True
    )

    importance["symbol"] = symbol

    output_path = (
        OUTPUT_DIR
        / f"{symbol.replace('.', '_')}_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False
    )

    print("\nTop 15 Features")
    print("-" * 70)

    print(
        importance[
            [
                "feature",
                "rf_importance",
                "permutation_importance",
                "combined_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    return importance


def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("FEATURE SELECTION EXPERIMENT")
    print("=" * 80)

    print(
        f"\nTotal features being analyzed: "
        f"{len(FEATURES)}"
    )

    stock_files = sorted(
        FEATURES_DIR.glob("*_features.csv")
    )

    all_results = []

    successful = 0
    failed = 0

    for filepath in stock_files:

        try:

            importance = analyze_stock(
                filepath
            )

            all_results.append(
                importance
            )

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

    # --------------------------------------------------
    # AGGREGATE ALL STOCKS
    # --------------------------------------------------

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        aggregate = (
            combined
            .groupby("feature")
            .agg(
                stocks_analyzed=(
                    "symbol",
                    "count"
                ),
                average_rf_importance=(
                    "rf_importance",
                    "mean"
                ),
                average_permutation_importance=(
                    "permutation_importance",
                    "mean"
                ),
                average_combined_score=(
                    "combined_score",
                    "mean"
                ),
            )
            .reset_index()
        )

        aggregate.sort_values(
            "average_combined_score",
            ascending=False,
            inplace=True
        )

        aggregate["rank"] = range(
            1,
            len(aggregate) + 1
        )

        aggregate_path = (
            OUTPUT_DIR
            / "aggregate_feature_importance.csv"
        )

        aggregate.to_csv(
            aggregate_path,
            index=False
        )

        print("\n")
        print("=" * 80)
        print("TOP FEATURES ACROSS ALL 20 STOCKS")
        print("=" * 80)

        print(
            aggregate[
                [
                    "rank",
                    "feature",
                    "average_rf_importance",
                    "average_permutation_importance",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        # --------------------------------------------------
        # FEATURE GROUP ANALYSIS
        # --------------------------------------------------

        def get_group(feature):

            if feature in STOCK_FEATURES:
                return "Stock / Technical"

            if feature in MARKET_FEATURES:
                return "Market Context"

            if feature in RELATIVE_FEATURES:
                return "Relative Strength"

            return "Other"

        aggregate["feature_group"] = (
            aggregate["feature"]
            .apply(get_group)
        )

        group_summary = (
            aggregate
            .groupby("feature_group")
            .agg(
                feature_count=(
                    "feature",
                    "count"
                ),
                avg_importance=(
                    "average_rf_importance",
                    "mean"
                ),
                avg_permutation=(
                    "average_permutation_importance",
                    "mean"
                ),
            )
            .sort_values(
                "avg_permutation",
                ascending=False
            )
        )

        print("\n")
        print("=" * 80)
        print("FEATURE GROUP IMPORTANCE")
        print("=" * 80)

        print(
            group_summary.to_string()
        )

        # --------------------------------------------------
        # TOP FEATURES
        # --------------------------------------------------

        top_features = aggregate.head(20)[
            [
                "rank",
                "feature",
                "feature_group",
                "average_rf_importance",
                "average_permutation_importance",
            ]
        ]

        top_features_path = (
            OUTPUT_DIR
            / "top_20_features.csv"
        )

        top_features.to_csv(
            top_features_path,
            index=False
        )

        print("\n")
        print(
            f"Saved aggregate results to:\n"
            f"{aggregate_path}"
        )

        print(
            f"\nSaved top features to:\n"
            f"{top_features_path}"
        )

    print("\n")
    print("=" * 80)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 80)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(stock_files)}")


if __name__ == "__main__":
    main()