from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

OUTPUT_DIR = Path(
    "data/models/v19_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

TRAIN_RATIO = 0.60


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Price / momentum
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    # Trend
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",

    # Momentum
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # Bollinger
    "BB_Position",

    # Volatility
    "Volatility_20D",
    "ATR_Percent",

    # Volume
    "Volume_Ratio",

    # Market context
    "NIFTY_Return_20D",
    "NIFTY_Return_5D",
    "NIFTY_SMA20_vs_SMA50",
    "NIFTY_SMA50_vs_SMA200",
    "NIFTY_Price_vs_SMA200",

    "BANKNIFTY_Return_20D",
    "BANKNIFTY_Return_5D",

    # Relative strength
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


# ============================================================
# FIND FEATURE FILES
# ============================================================

def get_feature_files():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    # Avoid accidental duplicates

    clean_files = []

    for file in files:

        name = file.name

        if (
            "_features_features"
            in name
        ):
            continue

        clean_files.append(
            file
        )

    return clean_files


# ============================================================
# SYMBOL
# ============================================================

def extract_symbol(
    file_path
):

    name = file_path.stem

    name = name.replace(
        "_features",
        ""
    )

    if name.endswith(
        "_NS"
    ):

        name = (
            name[:-3]
            + ".NS"
        )

    return name


# ============================================================
# LOAD DATA
# ============================================================

def load_all_data():

    files = (
        get_feature_files()
    )

    print(
        f"Feature files found: "
        f"{len(files)}"
    )

    all_data = []

    for file in files:

        symbol = extract_symbol(
            file
        )

        df = pd.read_csv(
            file
        )

        if "Date" not in df.columns:

            print(
                f"Skipping {file.name}: "
                f"no Date column"
            )

            continue

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        # Remove timezone

        if (
            hasattr(
                df["Date"].dt,
                "tz"
            )
            and df["Date"].dt.tz is not None
        ):

            df["Date"] = (
                df["Date"]
                .dt.tz_localize(
                    None
                )
            )

        df["Date"] = (
            df["Date"]
            .dt.normalize()
        )

        df["Symbol"] = symbol

        # ----------------------------------------------------
        # Forward return
        # ----------------------------------------------------

        df[
            "Future_Return_20D"
        ] = (

            df["Close"]
            .shift(-HORIZON)
            / df["Close"]
            - 1
        ) * 100

        # ----------------------------------------------------
        # Keep required columns
        # ----------------------------------------------------

        available_features = [
            f
            for f in FEATURES
            if f in df.columns
        ]

        required = [
            "Date",
            "Symbol",
            "Close",
            "Future_Return_20D",
        ]

        required += (
            available_features
        )

        df = df[
            required
        ].copy()

        df[
            available_features
        ] = (
            df[
                available_features
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        all_data.append(
            df
        )

        print(
            f"{symbol:<18} "
            f"{len(df):>5} rows"
        )

    if not all_data:

        raise RuntimeError(
            "No feature data found."
        )

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    return data


# ============================================================
# CROSS-SECTIONAL TARGET
# ============================================================

def create_ranking_target(
    data
):

    data = data.copy()

    # --------------------------------------------------------
    # Each day, compare every stock with the other stocks.
    #
    # Target = stock's forward return
    #          minus the daily average forward return.
    #
    # Positive = expected outperformer
    # Negative = expected underperformer
    # --------------------------------------------------------

    daily_mean = (
        data
        .groupby("Date")[
            "Future_Return_20D"
        ]
        .transform("mean")
    )

    data[
        "Relative_Return_20D"
    ] = (
        data[
            "Future_Return_20D"
        ]
        - daily_mean
    )

    return data


# ============================================================
# MODEL
# ============================================================

def build_models():

    models = {

        "Ridge":

        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    Ridge(
                        alpha=10
                    )
                ),
            ]
        ),

        "RandomForest":

        RandomForestRegressor(

            n_estimators=400,

            max_depth=8,

            min_samples_split=15,

            min_samples_leaf=5,

            random_state=42,

            n_jobs=-1,
        ),

        "GradientBoosting":

        GradientBoostingRegressor(

            n_estimators=250,

            learning_rate=0.03,

            max_depth=3,

            min_samples_leaf=10,

            random_state=42,
        ),
    }

    return models


# ============================================================
# RANK CORRELATION
# ============================================================

def rank_ic(
    actual,
    predicted
):

    a = pd.Series(
        actual
    )

    p = pd.Series(
        predicted
    )

    return (
        a.corr(
            p,
            method="spearman"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "V19 - CROSS-SECTIONAL RANKING MODEL"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    data = load_all_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    # ========================================================
    # TARGET
    # ========================================================

    data = (
        create_ranking_target(
            data
        )
    )

    # ========================================================
    # FEATURES AVAILABLE
    # ========================================================

    feature_columns = [
        f
        for f in FEATURES
        if f in data.columns
    ]

    print(
        f"Features available: "
        f"{len(feature_columns)}"
    )

    missing = [
        f
        for f in FEATURES
        if f not in data.columns
    ]

    if missing:

        print(
            "\nMissing features:"
        )

        for f in missing:
            print(
                f"  {f}"
            )

    # ========================================================
    # CLEAN
    # ========================================================

    data = data.dropna(
        subset=[
            "Relative_Return_20D"
        ]
    )

    data[
        feature_columns
    ] = (
        data[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Fill feature gaps using
    # cross-sectional daily median

    for feature in feature_columns:

        data[
            feature
        ] = (
            data
            .groupby("Date")[
                feature
            ]
            .transform(
                lambda x:
                x.fillna(
                    x.median()
                )
            )
        )

    data = data.dropna(
        subset=feature_columns
    )

    data = (
        data
        .sort_values(
            [
                "Date",
                "Symbol"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # DATE SPLIT
    # ========================================================

    dates = sorted(
        data[
            "Date"
        ].unique()
    )

    split_index = int(
        len(dates)
        * TRAIN_RATIO
    )

    train_end = dates[
        split_index - 1
    ]

    test_start = dates[
        split_index
    ]

    train = data[
        data["Date"]
        <= train_end
    ].copy()

    test = data[
        data["Date"]
        >= test_start
    ].copy()

    print("\n")
    print(
        "=" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Train: "
        f"{train['Date'].min().date()} "
        f"-> "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Test:  "
        f"{test['Date'].min().date()} "
        f"-> "
        f"{test['Date'].max().date()}"
    )

    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    # ========================================================
    # ARRAYS
    # ========================================================

    X_train = train[
        feature_columns
    ]

    y_train = train[
        "Relative_Return_20D"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "Relative_Return_20D"
    ]

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    models = build_models()

    predictions = []

    model_results = []

    for name, model in (
        models.items()
    ):

        print("\n")
        print(
            "-" * 70
        )

        print(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        test_copy = test[
            [
                "Date",
                "Symbol",
                "Future_Return_20D",
                "Relative_Return_20D",
            ]
        ].copy()

        test_copy[
            "Predicted_Relative_Return"
        ] = pred

        test_copy[
            "Model"
        ] = name

        predictions.append(
            test_copy
        )

        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        ic = rank_ic(
            y_test,
            pred
        )

        direction = (
            (
                np.sign(pred)
                ==
                np.sign(y_test)
            )
            .mean()
        )

        rmse = np.sqrt(
            np.mean(
                (
                    y_test
                    - pred
                ) ** 2
            )
        )

        print(
            f"RMSE: "
            f"{rmse:.4f}"
        )

        print(
            f"Rank IC: "
            f"{ic:.4f}"
        )

        print(
            f"Relative Direction: "
            f"{direction:.4f}"
        )

        # ----------------------------------------------------
        # Daily top-vs-bottom
        # ----------------------------------------------------

        test_copy[
            "Predicted_Rank"
        ] = (
            test_copy
            .groupby("Date")[
                "Predicted_Relative_Return"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        test_copy[
            "Actual_Rank"
        ] = (
            test_copy
            .groupby("Date")[
                "Relative_Return_20D"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        top = test_copy[
            test_copy[
                "Predicted_Rank"
            ] <= 5
        ]

        bottom = test_copy[
            test_copy[
                "Predicted_Rank"
            ] >= (
                test_copy
                .groupby("Date")[
                    "Symbol"
                ]
                .transform("count")
                - 4
            )
        ]

        top_return = (
            top[
                "Future_Return_20D"
            ].mean()
        )

        bottom_return = (
            bottom[
                "Future_Return_20D"
            ].mean()
        )

        spread = (
            top_return
            - bottom_return
        )

        print(
            f"Top-5 Avg Return: "
            f"{top_return:.3f}%"
        )

        print(
            f"Bottom-5 Avg Return: "
            f"{bottom_return:.3f}%"
        )

        print(
            f"Top-Bottom Spread: "
            f"{spread:.3f}%"
        )

        model_results.append({

            "Model":
                name,

            "RMSE":
                rmse,

            "Rank_IC":
                ic,

            "Relative_Direction":
                direction,

            "Top5_Return":
                top_return,

            "Bottom5_Return":
                bottom_return,

            "TopBottom_Spread":
                spread,
        })

    # ========================================================
    # SAVE
    # ========================================================

    all_predictions = pd.concat(
        predictions,
        ignore_index=True
    )

    results_df = pd.DataFrame(
        model_results
    )

    predictions_file = (
        OUTPUT_DIR
        / "v19_predictions.csv"
    )

    results_file = (
        OUTPUT_DIR
        / "v19_model_results.csv"
    )

    all_predictions.to_csv(
        predictions_file,
        index=False
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best = (
        results_df
        .sort_values(
            "Rank_IC",
            ascending=False
        )
        .iloc[0]
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "BEST RANKING MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: "
        f"{best['Model']}"
    )

    print(
        f"Rank IC: "
        f"{best['Rank_IC']:.4f}"
    )

    print(
        f"Top-5 Return: "
        f"{best['Top5_Return']:.3f}%"
    )

    print(
        f"Bottom-5 Return: "
        f"{best['Bottom5_Return']:.3f}%"
    )

    print(
        f"Top-Bottom Spread: "
        f"{best['TopBottom_Spread']:.3f}%"
    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        predictions_file.resolve()
    )

    print(
        results_file.resolve()
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "V19 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()