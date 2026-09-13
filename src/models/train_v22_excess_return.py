from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

OUTPUT_DIR = Path(
    "data/models/v22_excess_return"
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

    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    "SMA_20",
    "SMA_50",
    "EMA_20",

    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",

    "RSI_14",

    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    "BB_Position",

    "Volatility_20D",
    "ATR_Percent",

    "Volume_Ratio",

    "NIFTY_Return_5D",
    "NIFTY_Return_20D",

    "BANKNIFTY_Return_5D",
    "BANKNIFTY_Return_20D",

    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if (
        hasattr(
            df["Date"].dt,
            "tz"
        )
        and df["Date"].dt.tz is not None
    ):

        df["Date"] = (
            df["Date"]
            .dt.tz_localize(None)
        )

    df["Date"] = (
        df["Date"]
        .dt.normalize()
    )

    return df


# ============================================================
# FIND FILES
# ============================================================

def get_feature_files():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    return [
        f for f in files
        if "_features_features"
        not in f.name
    ]


def extract_symbol(path):

    name = path.stem

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

def load_data():

    datasets = []

    files = get_feature_files()

    print(
        f"Feature files found: "
        f"{len(files)}"
    )

    for file in files:

        symbol = extract_symbol(
            file
        )

        df = pd.read_csv(
            file
        )

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Stock future return
        # ----------------------------------------------------

        df[
            "Stock_Return_20D"
        ] = (

            df["Close"]
            .shift(-HORIZON)
            / df["Close"]
            - 1
        ) * 100

        # ----------------------------------------------------
        # NIFTY return
        #
        # Prefer the market return already present in the
        # feature dataset.
        # ----------------------------------------------------

        if (
            "NIFTY_Return_20D"
            not in df.columns
        ):

            print(
                f"WARNING: "
                f"{symbol} has no "
                f"NIFTY_Return_20D"
            )

            continue

        # ----------------------------------------------------
        # Excess return
        # ----------------------------------------------------

        df[
            "Excess_Return_20D"
        ] = (

            df[
                "Stock_Return_20D"
            ]
            - df[
                "NIFTY_Return_20D"
            ]
        )

        df["Symbol"] = symbol

        available_features = [
            f
            for f in FEATURES
            if f in df.columns
        ]

        required = [
            "Date",
            "Symbol",
            "Close",
            "Stock_Return_20D",
            "NIFTY_Return_20D",
            "Excess_Return_20D",
        ]

        required += (
            available_features
        )

        df = df[
            list(
                dict.fromkeys(
                    required
                )
            )
        ].copy()

        datasets.append(
            df
        )

        print(
            f"{symbol:<18}"
            f"{len(df):>5} rows"
        )

    if not datasets:

        raise RuntimeError(
            "No usable datasets found."
        )

    return pd.concat(
        datasets,
        ignore_index=True
    )


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

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


# ============================================================
# DAILY RANK IC
# ============================================================

def daily_rank_ic(
    df,
    prediction_column,
    target_column
):

    values = []

    for _, group in (
        df.groupby("Date")
    ):

        if len(group) < 5:
            continue

        correlation = (
            group[
                prediction_column
            ]
            .corr(
                group[
                    target_column
                ],
                method="spearman"
            )
        )

        if pd.notna(
            correlation
        ):

            values.append(
                correlation
            )

    if not values:

        return np.nan, np.nan

    return (
        float(
            np.mean(values)
        ),
        float(
            np.median(values)
        )
    )


# ============================================================
# TOP/BOTTOM ANALYSIS
# ============================================================

def ranking_analysis(
    df
):

    ranked = df.copy()

    ranked[
        "Rank"
    ] = (
        ranked
        .groupby("Date")[
            "Predicted_Excess_Return"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    ranked[
        "Stock_Count"
    ] = (
        ranked
        .groupby("Date")[
            "Symbol"
        ]
        .transform("count")
    )

    top = ranked[
        ranked["Rank"] <= 5
    ]

    bottom = ranked[
        ranked["Rank"]
        > (
            ranked["Stock_Count"]
            - 5
        )
    ]

    top_return = (
        top[
            "Excess_Return_20D"
        ].mean()
    )

    bottom_return = (
        bottom[
            "Excess_Return_20D"
        ].mean()
    )

    spread = (
        top_return
        - bottom_return
    )

    top_stock_return = (
        top[
            "Stock_Return_20D"
        ].mean()
    )

    top_win_rate = (
        top[
            "Excess_Return_20D"
        ] > 0
    ).mean()

    return {

        "Top5_Excess_Return":
            top_return,

        "Bottom5_Excess_Return":
            bottom_return,

        "TopBottom_Spread":
            spread,

        "Top5_Stock_Return":
            top_stock_return,

        "Top5_Excess_Win_Rate":
            top_win_rate,
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
        "V22 - EXCESS RETURN MODEL"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    data = load_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    feature_columns = [
        f
        for f in FEATURES
        if f in data.columns
    ]

    print(
        f"Features available: "
        f"{len(feature_columns)}"
    )

    # Fill missing features cross-sectionally

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
        subset=[
            "Excess_Return_20D"
        ]
        + feature_columns
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
    # SPLIT
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

    split_date = dates[
        split_index
    ]

    train = data[
        data["Date"]
        < split_date
    ].copy()

    test = data[
        data["Date"]
        >= split_date
    ].copy()

    print("\n")
    print(
        "=" * 70
    )

    print(
        "CHRONOLOGICAL SPLIT"
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
        "Excess_Return_20D"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "Excess_Return_20D"
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    models = build_models()

    all_predictions = []

    results = []

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

        prediction = model.predict(
            X_test
        )

        result = test[
            [
                "Date",
                "Symbol",
                "Stock_Return_20D",
                "NIFTY_Return_20D",
                "Excess_Return_20D",
            ]
        ].copy()

        result[
            "Predicted_Excess_Return"
        ] = prediction

        result[
            "Model"
        ] = name

        all_predictions.append(
            result
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        rmse = np.sqrt(
            np.mean(
                (
                    y_test
                    - prediction
                ) ** 2
            )
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        direction = (
            np.sign(
                prediction
            )
            ==
            np.sign(
                y_test
            )
        ).mean()

        # ----------------------------------------------------
        # Rank IC
        # ----------------------------------------------------

        mean_ic, median_ic = (
            daily_rank_ic(
                result,
                "Predicted_Excess_Return",
                "Excess_Return_20D"
            )
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        rank_stats = ranking_analysis(
            result
        )

        print(
            f"RMSE: "
            f"{rmse:.4f}%"
        )

        print(
            f"Relative Direction: "
            f"{direction * 100:.2f}%"
        )

        print(
            f"Mean Daily Rank IC: "
            f"{mean_ic:.4f}"
        )

        print(
            f"Median Daily Rank IC: "
            f"{median_ic:.4f}"
        )

        print(
            f"Top-5 Excess Return: "
            f"{rank_stats['Top5_Excess_Return']:.3f}%"
        )

        print(
            f"Bottom-5 Excess Return: "
            f"{rank_stats['Bottom5_Excess_Return']:.3f}%"
        )

        print(
            f"Top-Bottom Spread: "
            f"{rank_stats['TopBottom_Spread']:.3f}%"
        )

        print(
            f"Top-5 Stock Return: "
            f"{rank_stats['Top5_Stock_Return']:.3f}%"
        )

        print(
            f"Top-5 Excess Win Rate: "
            f"{rank_stats['Top5_Excess_Win_Rate'] * 100:.2f}%"
        )

        results.append({

            "Model":
                name,

            "RMSE":
                rmse,

            "Relative_Direction":
                direction,

            "Mean_Rank_IC":
                mean_ic,

            "Median_Rank_IC":
                median_ic,

            **rank_stats,
        })

    # ========================================================
    # SAVE
    # ========================================================

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    results_df = pd.DataFrame(
        results
    )

    predictions_file = (
        OUTPUT_DIR
        / "v22_predictions.csv"
    )

    results_file = (
        OUTPUT_DIR
        / "v22_model_results.csv"
    )

    predictions.to_csv(
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
            "Mean_Rank_IC",
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
        "BEST MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: "
        f"{best['Model']}"
    )

    print(
        f"Mean Rank IC: "
        f"{best['Mean_Rank_IC']:.4f}"
    )

    print(
        f"Top-Bottom Spread: "
        f"{best['TopBottom_Spread']:.3f}%"
    )

    # ========================================================
    # CURRENT PREDICTIONS
    # ========================================================

    best_model_name = (
        best["Model"]
    )

    current = predictions[
        predictions["Model"]
        == best_model_name
    ].copy()

    latest_date = (
        current["Date"].max()
    )

    current = (
        current[
            current["Date"]
            == latest_date
        ]
        .sort_values(
            "Predicted_Excess_Return",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    current[
        "Rank"
    ] = np.arange(
        1,
        len(current) + 1
    )

    current_file = (
        OUTPUT_DIR
        / "current_ranking.csv"
    )

    current.to_csv(
        current_file,
        index=False
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "CURRENT STOCK RANKING"
    )

    print(
        "=" * 70
    )

    print(
        current[
            [
                "Rank",
                "Symbol",
                "Predicted_Excess_Return",
                "NIFTY_Return_20D",
            ]
        ].to_string(
            index=False
        )
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

    print(
        current_file.resolve()
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "V22 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()