from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "models"
    / "regime_walk_forward"
)

OUTPUT_FILE = (
    MODEL_DIR
    / "regime_walk_forward_predictions.csv"
)

HORIZON = 20

INITIAL_TRAIN_RATIO = 0.60

RETRAIN_EVERY = 60

BUY_THRESHOLD = 5.0

SELL_THRESHOLD = -5.0

MIN_BUY_PROB = 0.60

MIN_SELL_PROB = 0.60

MIN_EXPECTED_RETURN = 2.0

MAX_EXPECTED_RETURN = -2.0


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
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
    "NIFTY_Return_1D",
    "NIFTY_Return_5D",
    "NIFTY_Return_20D",

    "NIFTY_Price_vs_SMA20",
    "NIFTY_Price_vs_SMA50",
    "NIFTY_Price_vs_SMA200",

    "NIFTY_SMA20_vs_SMA50",
    "NIFTY_SMA50_vs_SMA200",

    "NIFTY_Volatility_5D",
    "NIFTY_Volatility_20D",

    "Market_Regime_Score",

    "Relative_Strength_1D",
    "Relative_Strength_5D",
    "Relative_Strength_20D",
]


FEATURES = BASE_FEATURES + MARKET_FEATURES


# ============================================================
# SYMBOL EXTRACTION
# ============================================================

def extract_symbol(file_path):

    filename = file_path.stem

    # Handle:
    #
    # ADANIENT_NS_features_regime
    # ADANIENT_NS_features_features_regime
    #
    filename = filename.replace(
        "_features_features_regime",
        ""
    )

    filename = filename.replace(
        "_features_regime",
        ""
    )

    filename = filename.replace(
        "_features",
        ""
    )

    # Convert:
    #
    # ADANIENT_NS
    #
    # to:
    #
    # ADANIENT.NS

    if filename.endswith("_NS"):

        filename = (
            filename[:-3]
            + ".NS"
        )

    return filename


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    symbol = extract_symbol(file_path)

    df = pd.read_csv(file_path)

    if "Date" not in df.columns:

        raise ValueError(
            f"Date column missing in "
            f"{file_path.name}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    missing = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{symbol}: Missing features: "
            f"{missing}"
        )

    return (
        symbol,
        df.reset_index(drop=True)
    )


# ============================================================
# TARGETS
# ============================================================

def prepare_targets(df):

    df = df.copy()

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    df["BUY_TARGET"] = (
        df["Future_Return_20D"]
        > BUY_THRESHOLD
    ).astype(int)

    df["SELL_TARGET"] = (
        df["Future_Return_20D"]
        < SELL_THRESHOLD
    ).astype(int)

    return df


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(df):

    X = df[
        FEATURES
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.ffill().bfill()

    return X


# ============================================================
# MODELS
# ============================================================

def create_buy_model():

    return RandomForestClassifier(

        n_estimators=300,

        max_depth=10,

        min_samples_split=10,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )


def create_sell_model():

    return RandomForestClassifier(

        n_estimators=300,

        max_depth=10,

        min_samples_split=10,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )


def create_return_model():

    return RandomForestRegressor(

        n_estimators=300,

        max_depth=10,

        min_samples_split=10,

        min_samples_leaf=3,

        random_state=42,

        n_jobs=-1
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(row):

    score = 0

    if row["Close"] > row["SMA_20"]:

        score += 1

    else:

        score -= 1


    if row["SMA_20"] > row["SMA_50"]:

        score += 1

    else:

        score -= 1


    if row["MACD"] > row["MACD_Signal"]:

        score += 1

    else:

        score -= 1


    if (
        50
        <= row["RSI_14"]
        <= 70
    ):

        score += 1

    elif row["RSI_14"] < 30:

        score += 1

    elif row["RSI_14"] > 70:

        score -= 1


    return float(score)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    buy_probability,
    sell_probability,
    expected_return,
    technical_score,
    market_regime_score
):

    technical_component = np.clip(
        technical_score / 4.0,
        -1,
        1
    )

    market_component = np.clip(
        market_regime_score,
        -1,
        1
    )

    expected_component = np.clip(
        expected_return / 10.0,
        -1,
        1
    )

    final_score = (

        buy_probability * 0.40

        - sell_probability * 0.25

        + technical_component * 0.20

        + market_component * 0.10

        + expected_component * 0.05
    )


    # BUY

    if (

        buy_probability
        >= MIN_BUY_PROB

        and expected_return
        >= MIN_EXPECTED_RETURN

        and final_score
        >= 0.20

    ):

        signal = "BUY"


    # SELL

    elif (

        sell_probability
        >= MIN_SELL_PROB

        and expected_return
        <= MAX_EXPECTED_RETURN

        and final_score
        <= -0.20

    ):

        signal = "SELL"


    else:

        signal = "HOLD"


    return (
        signal,
        final_score
    )


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_stock(
    symbol,
    df
):

    df = prepare_targets(df)

    valid_df = df[
        df["Future_Return_20D"]
        .notna()
    ].copy()

    if len(valid_df) < 300:

        print(
            f"WARNING {symbol}: "
            f"Not enough rows"
        )

        return []


    initial_train_size = int(
        len(valid_df)
        * INITIAL_TRAIN_RATIO
    )

    predictions = []

    test_start = (
        initial_train_size
    )

    fold_number = 0


    while test_start < len(valid_df):

        fold_number += 1

        test_end = min(

            test_start
            + RETRAIN_EVERY,

            len(valid_df)
        )


        test_df = valid_df.iloc[
            test_start:test_end
        ].copy()


        # ----------------------------------------------------
        # 20 DAY EMBARGO
        # ----------------------------------------------------

        train_end = (
            test_start
            - HORIZON
        )


        if train_end <= 0:

            test_start = test_end

            continue


        train_df = valid_df.iloc[
            :train_end
        ].copy()


        if len(train_df) < 200:

            test_start = test_end

            continue


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X_train = clean_features(
            train_df
        )

        X_test = clean_features(
            test_df
        )


        y_buy = (
            train_df["BUY_TARGET"]
        )

        y_sell = (
            train_df["SELL_TARGET"]
        )

        y_return = (
            train_df["Future_Return_20D"]
        )


        # ----------------------------------------------------
        # REMOVE INVALID ROWS
        # ----------------------------------------------------

        valid_train = (

            X_train.notna()
            .all(axis=1)

            & y_buy.notna()

            & y_sell.notna()

            & y_return.notna()
        )


        X_train = X_train.loc[
            valid_train
        ]

        y_buy = y_buy.loc[
            valid_train
        ]

        y_sell = y_sell.loc[
            valid_train
        ]

        y_return = y_return.loc[
            valid_train
        ]


        valid_test = (
            X_test.notna()
            .all(axis=1)
        )


        X_test = X_test.loc[
            valid_test
        ]

        test_df = test_df.loc[
            valid_test
        ].copy()


        if len(X_train) < 100:

            test_start = test_end

            continue


        if len(X_test) == 0:

            test_start = test_end

            continue


        if y_buy.nunique() < 2:

            test_start = test_end

            continue


        if y_sell.nunique() < 2:

            test_start = test_end

            continue


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        buy_model = (
            create_buy_model()
        )

        sell_model = (
            create_sell_model()
        )

        return_model = (
            create_return_model()
        )


        buy_model.fit(
            X_train,
            y_buy
        )

        sell_model.fit(
            X_train,
            y_sell
        )

        return_model.fit(
            X_train,
            y_return
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        buy_probability = (
            buy_model
            .predict_proba(X_test)[:, 1]
        )


        sell_probability = (
            sell_model
            .predict_proba(X_test)[:, 1]
        )


        expected_return = (
            return_model
            .predict(X_test)
        )


        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        for i, (_, row) in enumerate(
            test_df.iterrows()
        ):

            buy_prob = float(
                buy_probability[i]
            )

            sell_prob = float(
                sell_probability[i]
            )

            exp_return = float(
                expected_return[i]
            )


            technical_score = (
                calculate_technical_score(
                    row
                )
            )


            market_regime_score = float(
                row[
                    "Market_Regime_Score"
                ]
            )


            signal, final_score = (
                generate_signal(

                    buy_probability=buy_prob,

                    sell_probability=sell_prob,

                    expected_return=exp_return,

                    technical_score=technical_score,

                    market_regime_score=market_regime_score
                )
            )


            predictions.append({

                "Symbol":
                    symbol,

                "Date":
                    row["Date"],

                "Close":
                    row["Close"],

                "Buy_Probability":
                    buy_prob,

                "Sell_Probability":
                    sell_prob,

                "Expected_Return_20D":
                    exp_return,

                "Technical_Score":
                    technical_score,

                "Market_Regime":
                    row.get(
                        "Market_Regime",
                        "UNKNOWN"
                    ),

                "Market_Regime_Score":
                    market_regime_score,

                "Relative_Strength_1D":
                    row[
                        "Relative_Strength_1D"
                    ],

                "Relative_Strength_5D":
                    row[
                        "Relative_Strength_5D"
                    ],

                "Relative_Strength_20D":
                    row[
                        "Relative_Strength_20D"
                    ],

                "Final_Score":
                    final_score,

                "Signal":
                    signal,

                "Actual_Return_20D":
                    row[
                        "Future_Return_20D"
                    ],

                "BUY_TARGET":
                    row["BUY_TARGET"],

                "SELL_TARGET":
                    row["SELL_TARGET"],

                "Fold":
                    fold_number
            })


        print(

            f"  Fold {fold_number}: "

            f"train={len(train_df):,} "

            f"test={len(test_df):,} "

            f"predictions={len(test_df):,}"
        )


        test_start = test_end


    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "REGIME-AWARE TRUE CLEAN "
        "WALK-FORWARD MODEL"
    )

    print("=" * 70)


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # IMPORTANT:
    #
    # ONLY load regime files.
    #
    # Do NOT load the original feature files.
    # ========================================================

    files = sorted(
        FEATURE_DIR.glob(
            "*_features_regime.csv"
        )
    )


    if not files:

        print(
            "\nERROR: "
            "No *_features_regime.csv "
            "files found."
        )

        print(
            f"Folder:\n{FEATURE_DIR}"
        )

        return


    print(
        f"\nFound {len(files)} "
        "regime feature files."
    )


    # Safety check

    if len(files) != 20:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 20 regime files."
        )

        print(
            "Found:",
            len(files)
        )

        print(
            "\nFiles detected:"
        )

        for f in files:

            print(
                " ",
                f.name
            )


    print(
        f"\nHorizon: "
        f"{HORIZON} trading days"
    )

    print(
        f"Initial training: "
        f"{INITIAL_TRAIN_RATIO:.0%}"
    )

    print(
        f"Retrain every: "
        f"{RETRAIN_EVERY} days"
    )

    print()


    all_predictions = []

    successful = 0


    # ========================================================
    # PROCESS STOCKS
    # ========================================================

    for file_path in files:

        try:

            symbol, df = load_data(
                file_path
            )


            print(
                "\n"
                + "-" * 60
            )

            print(
                f"FILE: "
                f"{file_path.name}"
            )

            print(
                f"SYMBOL: "
                f"{symbol}"
            )

            print(
                f"Rows: "
                f"{len(df):,}"
            )


            predictions = (
                walk_forward_stock(
                    symbol,
                    df
                )
            )


            if predictions:

                all_predictions.extend(
                    predictions
                )

                successful += 1


                print(
                    f"SUCCESS: "
                    f"{len(predictions):,} "
                    f"predictions"
                )

            else:

                print(
                    "WARNING: "
                    "No predictions generated"
                )


        except Exception as e:

            print(
                f"ERROR "
                f"{file_path.name}: "
                f"{e}"
            )


    # ========================================================
    # RESULTS
    # ========================================================

    if not all_predictions:

        print(
            "\nERROR: "
            "No predictions generated."
        )

        return


    results = pd.DataFrame(
        all_predictions
    )


    results["Date"] = pd.to_datetime(
        results["Date"]
    )


    results = results.sort_values(
        [
            "Date",
            "Symbol"
        ]
    ).reset_index(
        drop=True
    )


    results.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"Stocks processed: "
        f"{successful}/{len(files)}"
    )


    print(
        f"Predictions: "
        f"{len(results):,}"
    )


    print(
        f"Date range: "
        f"{results['Date'].min().date()} "
        f"-> "
        f"{results['Date'].max().date()}"
    )


    print(
        "\nSignal distribution:"
    )

    print(
        results[
            "Signal"
        ].value_counts().to_string()
    )


    print(
        "\nAverage probabilities:"
    )

    print(
        f"BUY: "
        f"{results['Buy_Probability'].mean():.4f}"
    )

    print(
        f"SELL: "
        f"{results['Sell_Probability'].mean():.4f}"
    )


    print(
        f"\nExpected 20D return: "
        f"{results['Expected_Return_20D'].mean():.4f}%"
    )


    print(
        f"Actual 20D return: "
        f"{results['Actual_Return_20D'].mean():.4f}%"
    )


    print(
        "\nMarket regimes:"
    )

    print(
        results[
            "Market_Regime"
        ].value_counts().to_string()
    )


    # ========================================================
    # BUY ANALYSIS
    # ========================================================

    buy = results[
        results["Signal"] == "BUY"
    ]


    if len(buy) > 0:

        print(
            "\nBUY statistics:"
        )

        print(
            f"Count: "
            f"{len(buy):,}"
        )

        print(
            f"Average actual return: "
            f"{buy['Actual_Return_20D'].mean():.2f}%"
        )

        print(
            f"Win rate: "
            f"{(
                buy["Actual_Return_20D"]
                > 0
            ).mean():.2%}"
        )


    # ========================================================
    # SELL ANALYSIS
    # ========================================================

    sell = results[
        results["Signal"] == "SELL"
    ]


    if len(sell) > 0:

        print(
            "\nSELL statistics:"
        )

        print(
            f"Count: "
            f"{len(sell):,}"
        )

        print(
            f"Average actual return: "
            f"{sell['Actual_Return_20D'].mean():.2f}%"
        )

        print(
            f"Correct SELL rate: "
            f"{(
                sell["Actual_Return_20D"]
                < 0
            ).mean():.2%}"
        )


    # ========================================================
    # SYMBOL VALIDATION
    # ========================================================

    print(
        "\nSymbol validation:"
    )


    symbols = sorted(
        results[
            "Symbol"
        ].unique()
    )


    print(
        symbols
    )


    bad_symbols = [

        s
        for s in symbols

        if not s.endswith(".NS")
    ]


    if bad_symbols:

        print(
            "\nWARNING: "
            "Invalid symbols detected:"
        )

        print(
            bad_symbols
        )

    else:

        print(
            "\nALL SYMBOLS CORRECT."
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()