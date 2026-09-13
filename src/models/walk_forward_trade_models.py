import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(BASE_DIR)

from src.data.stock_universe import INDIAN_STOCKS


# ============================================================
# SETTINGS
# ============================================================

HORIZON = 20

INITIAL_TRAIN_RATIO = 0.60

TEST_WINDOW = 60

N_ESTIMATORS = 200

RANDOM_STATE = 42

BUY_THRESHOLD = 0.50

SELL_THRESHOLD = 0.50


# ============================================================
# DIRECTORIES
# ============================================================

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "walk_forward_trade_models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(symbol):

    file_path = os.path.join(
        FEATURE_DIR,
        f"{symbol.replace('.', '_')}_features.csv"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Feature file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

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

    if date_column != "date":

        df.rename(
            columns={
                date_column: "date"
            },
            inplace=True
        )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Create future return
    # --------------------------------------------------------

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    df["Buy_Target"] = (
        df["Future_Return_20D"] > 5
    ).astype(int)

    df["Sell_Target"] = (
        df["Future_Return_20D"] < -5
    ).astype(int)

    return df


# ============================================================
# FEATURES
# ============================================================

def get_features(df):

    excluded = {

        "date",

        "Future_Return_5D",

        "Future_Return_20D",

        "Buy_Target",

        "Sell_Target",

        "Target",

        "Return_Target",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return features


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=10,

        min_samples_split=10,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_stock(symbol):

    print("\n")
    print("=" * 70)
    print(f"WALK-FORWARD: {symbol}")
    print("=" * 70)

    df = load_data(
        symbol
    )

    features = get_features(
        df
    )

    # --------------------------------------------------------
    # Remove rows without complete feature data
    # --------------------------------------------------------

    df = df.dropna(
        subset=features
    ).reset_index(
        drop=True
    )

    total_rows = len(df)

    initial_train_end = int(
        total_rows * INITIAL_TRAIN_RATIO
    )

    predictions = []

    actual_buy = []

    actual_sell = []

    actual_returns = []

    buy_probabilities = []

    sell_probabilities = []

    prediction_dates = []

    # ========================================================
    # WALK FORWARD
    # ========================================================

    test_start = initial_train_end

    while test_start < total_rows:

        test_end = min(
            test_start + TEST_WINDOW,
            total_rows
        )

        # ----------------------------------------------------
        # CRITICAL:
        # Training labels need 20 future trading days.
        #
        # Therefore training must stop 20 rows before the
        # test period. Otherwise a training label could use
        # prices from inside the test period.
        # ----------------------------------------------------

        safe_train_end = (
            test_start - HORIZON
        )

        if safe_train_end <= 0:

            break

        train_df = df.iloc[
            :safe_train_end
        ].copy()

        test_df = df.iloc[
            test_start:test_end
        ].copy()

        # ----------------------------------------------------
        # Drop rows where future return is unavailable
        # ----------------------------------------------------

        train_df = train_df.dropna(
            subset=[
                "Future_Return_20D"
            ]
        )

        test_df = test_df.dropna(
            subset=[
                "Future_Return_20D"
            ]
        )

        if len(train_df) < 100:

            print(
                "Skipping window: insufficient training data"
            )

            break

        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        X_train = train_df[
            features
        ]

        y_buy_train = train_df[
            "Buy_Target"
        ]

        y_sell_train = train_df[
            "Sell_Target"
        ]

        # ----------------------------------------------------
        # Test data
        # ----------------------------------------------------

        X_test = test_df[
            features
        ]

        # ----------------------------------------------------
        # Train BUY model
        # ----------------------------------------------------

        buy_model = train_model(
            X_train,
            y_buy_train
        )

        # ----------------------------------------------------
        # Train SELL model
        # ----------------------------------------------------

        sell_model = train_model(
            X_train,
            y_sell_train
        )

        # ----------------------------------------------------
        # Predict probabilities
        # ----------------------------------------------------

        buy_probability = (
            buy_model
            .predict_proba(
                X_test
            )[:, 1]
        )

        sell_probability = (
            sell_model
            .predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        predictions.extend(
            [
                "BUY"
                if buy_probability[i] >= BUY_THRESHOLD
                and buy_probability[i] > sell_probability[i]

                else "SELL"
                if sell_probability[i] >= SELL_THRESHOLD
                and sell_probability[i] > buy_probability[i]

                else "HOLD"

                for i in range(
                    len(test_df)
                )
            ]
        )

        actual_buy.extend(
            test_df[
                "Buy_Target"
            ].tolist()
        )

        actual_sell.extend(
            test_df[
                "Sell_Target"
            ].tolist()
        )

        actual_returns.extend(
            test_df[
                "Future_Return_20D"
            ].tolist()
        )

        buy_probabilities.extend(
            buy_probability.tolist()
        )

        sell_probabilities.extend(
            sell_probability.tolist()
        )

        prediction_dates.extend(
            test_df[
                "date"
            ].tolist()
        )

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        test_start = test_end

    # ========================================================
    # RESULTS
    # ========================================================

    if len(predictions) == 0:

        return None, None

    results_df = pd.DataFrame({

        "symbol": symbol,

        "date": prediction_dates,

        "buy_probability": buy_probabilities,

        "sell_probability": sell_probabilities,

        "prediction": predictions,

        "actual_buy": actual_buy,

        "actual_sell": actual_sell,

        "future_return_20d": actual_returns,

    })

    # ========================================================
    # METRICS
    # ========================================================

    buy_accuracy = accuracy_score(
        actual_buy,
        [
            1
            if p >= BUY_THRESHOLD
            else 0
            for p in buy_probabilities
        ]
    )

    sell_accuracy = accuracy_score(
        actual_sell,
        [
            1
            if p >= SELL_THRESHOLD
            else 0
            for p in sell_probabilities
        ]
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    try:

        buy_auc = roc_auc_score(
            actual_buy,
            buy_probabilities
        )

    except ValueError:

        buy_auc = np.nan

    try:

        sell_auc = roc_auc_score(
            actual_sell,
            sell_probabilities
        )

    except ValueError:

        sell_auc = np.nan

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    buy_baseline = max(
        np.mean(actual_buy),
        1 - np.mean(actual_buy)
    )

    sell_baseline = max(
        np.mean(actual_sell),
        1 - np.mean(actual_sell)
    )

    # --------------------------------------------------------
    # Probability / return correlation
    # --------------------------------------------------------

    if (
        np.std(buy_probabilities) > 0
        and np.std(actual_returns) > 0
    ):

        buy_return_corr = np.corrcoef(
            buy_probabilities,
            actual_returns
        )[0, 1]

    else:

        buy_return_corr = np.nan

    if (
        np.std(sell_probabilities) > 0
        and np.std(actual_returns) > 0
    ):

        sell_return_corr = np.corrcoef(
            sell_probabilities,
            actual_returns
        )[0, 1]

    else:

        sell_return_corr = np.nan

    # ========================================================
    # HIGH-CONFIDENCE ANALYSIS
    # ========================================================

    buy_high = results_df[
        results_df[
            "buy_probability"
        ] >= 0.70
    ]

    sell_high = results_df[
        results_df[
            "sell_probability"
        ] >= 0.70
    ]

    if len(buy_high) > 0:

        high_buy_avg_return = (
            buy_high[
                "future_return_20d"
            ].mean()
        )

        high_buy_win_rate = (
            buy_high[
                "actual_buy"
            ].mean()
        )

    else:

        high_buy_avg_return = np.nan

        high_buy_win_rate = np.nan

    if len(sell_high) > 0:

        high_sell_avg_return = (
            sell_high[
                "future_return_20d"
            ].mean()
        )

        high_sell_win_rate = (
            sell_high[
                "actual_sell"
            ].mean()
        )

    else:

        high_sell_avg_return = np.nan

        high_sell_win_rate = np.nan

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "symbol": symbol,

        "predictions": len(
            results_df
        ),

        "buy_rate": np.mean(
            actual_buy
        ),

        "sell_rate": np.mean(
            actual_sell
        ),

        "buy_accuracy": buy_accuracy,

        "buy_baseline": buy_baseline,

        "buy_auc": buy_auc,

        "sell_accuracy": sell_accuracy,

        "sell_baseline": sell_baseline,

        "sell_auc": sell_auc,

        "buy_return_correlation": buy_return_corr,

        "sell_return_correlation": sell_return_corr,

        "high_buy_predictions": len(
            buy_high
        ),

        "high_buy_avg_return": high_buy_avg_return,

        "high_buy_win_rate": high_buy_win_rate,

        "high_sell_predictions": len(
            sell_high
        ),

        "high_sell_avg_return": high_sell_avg_return,

        "high_sell_win_rate": high_sell_win_rate,
    }

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Predictions: {summary['predictions']}"
    )

    print(
        f"BUY AUC: "
        f"{summary['buy_auc']:.4f}"
    )

    print(
        f"BUY Accuracy: "
        f"{summary['buy_accuracy']:.4f}"
    )

    print(
        f"BUY Baseline: "
        f"{summary['buy_baseline']:.4f}"
    )

    print(
        f"SELL AUC: "
        f"{summary['sell_auc']:.4f}"
    )

    print(
        f"SELL Accuracy: "
        f"{summary['sell_accuracy']:.4f}"
    )

    print(
        f"SELL Baseline: "
        f"{summary['sell_baseline']:.4f}"
    )

    print(
        f"BUY probability/return correlation: "
        f"{summary['buy_return_correlation']:.4f}"
    )

    print(
        f"SELL probability/return correlation: "
        f"{summary['sell_return_correlation']:.4f}"
    )

    print(
        f"BUY >=70% predictions: "
        f"{summary['high_buy_predictions']}"
    )

    print(
        f"BUY >=70% average return: "
        f"{summary['high_buy_avg_return']:+.2f}%"
    )

    print(
        f"BUY >=70% meaningful BUY rate: "
        f"{summary['high_buy_win_rate']:.2%}"
    )

    print(
        f"SELL >=70% predictions: "
        f"{summary['high_sell_predictions']}"
    )

    print(
        f"SELL >=70% average return: "
        f"{summary['high_sell_avg_return']:+.2f}%"
    )

    print(
        f"SELL >=70% meaningful SELL rate: "
        f"{summary['high_sell_win_rate']:.2%}"
    )

    return (
        summary,
        results_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    summaries = []

    all_predictions = []

    print("\n")
    print("=" * 80)
    print("20-DAY TRADE MODEL WALK-FORWARD VALIDATION")
    print("=" * 80)

    print(
        f"\nHorizon: {HORIZON} trading days"
    )

    print(
        f"Initial training: "
        f"{INITIAL_TRAIN_RATIO:.0%}"
    )

    print(
        f"Test window: "
        f"{TEST_WINDOW} days"
    )

    print(
        "\nIMPORTANT: "
        "Training labels are stopped 20 days before "
        "each test window to prevent look-ahead leakage."
    )

    # ========================================================
    # ALL STOCKS
    # ========================================================

    for symbol in INDIAN_STOCKS:

        try:

            summary, predictions = (
                walk_forward_stock(
                    symbol
                )
            )

            if summary is not None:

                summaries.append(
                    summary
                )

                all_predictions.append(
                    predictions
                )

        except Exception as e:

            print(
                f"\nERROR: {symbol}"
            )

            print(
                str(e)
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if not summaries:

        print(
            "\nNo validation results generated."
        )

        return

    summary_df = pd.DataFrame(
        summaries
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "walk_forward_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_file = os.path.join(
        OUTPUT_DIR,
        "walk_forward_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\n")
    print("=" * 90)
    print("OVERALL WALK-FORWARD RESULTS")
    print("=" * 90)

    print(
        f"\nStocks tested: "
        f"{len(summary_df)}"
    )

    print(
        f"Total predictions: "
        f"{len(predictions_df)}"
    )

    print(
        f"\nAverage BUY AUC: "
        f"{summary_df['buy_auc'].mean():.4f}"
    )

    print(
        f"Median BUY AUC: "
        f"{summary_df['buy_auc'].median():.4f}"
    )

    print(
        f"Average SELL AUC: "
        f"{summary_df['sell_auc'].mean():.4f}"
    )

    print(
        f"Median SELL AUC: "
        f"{summary_df['sell_auc'].median():.4f}"
    )

    print(
        f"\nAverage BUY Accuracy: "
        f"{summary_df['buy_accuracy'].mean():.4f}"
    )

    print(
        f"Average BUY Baseline: "
        f"{summary_df['buy_baseline'].mean():.4f}"
    )

    print(
        f"Average SELL Accuracy: "
        f"{summary_df['sell_accuracy'].mean():.4f}"
    )

    print(
        f"Average SELL Baseline: "
        f"{summary_df['sell_baseline'].mean():.4f}"
    )

    print(
        f"\nAverage BUY return correlation: "
        f"{summary_df['buy_return_correlation'].mean():.4f}"
    )

    print(
        f"Average SELL return correlation: "
        f"{summary_df['sell_return_correlation'].mean():.4f}"
    )

    # ========================================================
    # STOCK TABLE
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STOCK-LEVEL RESULTS")
    print("=" * 110)

    display_columns = [

        "symbol",

        "predictions",

        "buy_auc",

        "sell_auc",

        "buy_accuracy",

        "buy_baseline",

        "sell_accuracy",

        "sell_baseline",

        "high_buy_predictions",

        "high_buy_avg_return",

        "high_sell_predictions",

        "high_sell_avg_return",
    ]

    display_df = summary_df[
        display_columns
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # BEST STOCKS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BEST BUY MODELS")
    print("=" * 70)

    print(
        summary_df[
            [
                "symbol",
                "buy_auc",
                "high_buy_avg_return",
                "high_buy_win_rate",
            ]
        ]
        .sort_values(
            "buy_auc",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print("BEST SELL MODELS")
    print("=" * 70)

    print(
        summary_df[
            [
                "symbol",
                "sell_auc",
                "high_sell_avg_return",
                "high_sell_win_rate",
            ]
        ]
        .sort_values(
            "sell_auc",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # FILES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        summary_file
    )

    print(
        predictions_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()