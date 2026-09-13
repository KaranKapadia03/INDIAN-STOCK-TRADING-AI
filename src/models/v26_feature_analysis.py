from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

V24_PREDICTIONS = Path(
    "data/models/v24_walk_forward/v24_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/models/v26_feature_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

MIN_TRAIN_DAYS = 500

RANDOM_STATE = 42


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
# DATE
# ============================================================

def normalize_dates(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    try:

        if df["Date"].dt.tz is not None:

            df["Date"] = (
                df["Date"]
                .dt.tz_localize(None)
            )

    except Exception:
        pass

    df["Date"] = (
        df["Date"]
        .dt.normalize()
    )

    return df


# ============================================================
# SYMBOL
# ============================================================

def extract_symbol(path):

    name = path.stem

    name = name.replace(
        "_features",
        ""
    )

    if name.endswith("_NS"):

        name = (
            name[:-3]
            + ".NS"
        )

    return name


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_data():

    datasets = []

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

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
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        # Future stock return

        df[
            "Stock_Return_20D"
        ] = (

            df["Close"].shift(-HORIZON)
            / df["Close"]
            - 1
        ) * 100

        if (
            "NIFTY_Return_20D"
            not in df.columns
        ):

            continue

        # Excess return

        df[
            "Excess_Return_20D"
        ] = (

            df["Stock_Return_20D"]
            - df["NIFTY_Return_20D"]
        )

        df["Symbol"] = symbol

        available = [
            f
            for f in FEATURES
            if f in df.columns
        ]

        required = [
            "Date",
            "Symbol",
            "Excess_Return_20D",
        ]

        required += available

        df = df[
            list(
                dict.fromkeys(
                    required
                )
            )
        ]

        datasets.append(
            df
        )

    if not datasets:

        raise RuntimeError(
            "No datasets found."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return data


# ============================================================
# PREPARE
# ============================================================

def prepare_data(data):

    features = [
        f
        for f in FEATURES
        if f in data.columns
    ]

    for feature in features:

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
        ] + features
    )

    return (
        data
        .sort_values(
            [
                "Date",
                "Symbol"
            ]
        )
        .reset_index(drop=True),
        features
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(
    data,
    features
):

    rows = []

    for feature in features:

        correlation = (
            data[
                feature
            ].corr(
                data[
                    "Excess_Return_20D"
                ]
            )
        )

        rank_correlation = (
            data[
                feature
            ].corr(
                data[
                    "Excess_Return_20D"
                ],
                method="spearman"
            )
        )

        rows.append({

            "Feature":
                feature,

            "Pearson_Correlation":
                correlation,

            "Spearman_Correlation":
                rank_correlation,

            "Absolute_Pearson":
                abs(correlation),

            "Absolute_Spearman":
                abs(rank_correlation),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Absolute_Spearman",
            ascending=False
        )
    )


# ============================================================
# RANDOM FOREST IMPORTANCE
# ============================================================

def random_forest_importance(
    data,
    features
):

    dates = sorted(
        data["Date"].unique()
    )

    split_index = int(
        len(dates) * 0.60
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

    X_train = train[
        features
    ]

    y_train = train[
        "Excess_Return_20D"
    ]

    X_test = test[
        features
    ]

    y_test = test[
        "Excess_Return_20D"
    ]

    model = RandomForestRegressor(

        n_estimators=400,

        max_depth=8,

        min_samples_split=15,

        min_samples_leaf=5,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    print("\n")
    print(
        "Training feature-importance model..."
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Built-in importance
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "Feature":
            features,

        "RF_Importance":
            model.feature_importances_,
    })

    # --------------------------------------------------------
    # Permutation importance on unseen test data
    # --------------------------------------------------------

    print(
        "Calculating permutation importance..."
    )

    permutation = (
        permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_mean_squared_error"
        )
    )

    permutation_df = pd.DataFrame({

        "Feature":
            features,

        "Permutation_Importance":
            permutation.importances_mean,

        "Permutation_Std":
            permutation.importances_std,
    })

    result = importance.merge(
        permutation_df,
        on="Feature"
    )

    result[
        "Absolute_Permutation"
    ] = (
        result[
            "Permutation_Importance"
        ].abs()
    )

    result = (
        result
        .sort_values(
            "RF_Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        result,
        train,
        test,
        model
    )


# ============================================================
# TIME STABILITY
# ============================================================

def time_stability(
    data,
    features
):

    dates = sorted(
        data["Date"].unique()
    )

    windows = []

    # Divide history into 4 chronological sections.

    n = len(dates)

    boundaries = [

        (0, int(n * 0.25)),

        (int(n * 0.25), int(n * 0.50)),

        (int(n * 0.50), int(n * 0.75)),

        (int(n * 0.75), n),
    ]

    for window_number, (
        start,
        end
    ) in enumerate(
        boundaries,
        start=1
    ):

        start_date = dates[start]
        end_date = dates[end - 1]

        window = data[
            (
                data["Date"]
                >= start_date
            )
            &
            (
                data["Date"]
                <= end_date
            )
        ].copy()

        for feature in features:

            correlation = (
                window[
                    feature
                ].corr(
                    window[
                        "Excess_Return_20D"
                    ],
                    method="spearman"
                )
            )

            windows.append({

                "Window":
                    window_number,

                "Start":
                    start_date,

                "End":
                    end_date,

                "Feature":
                    feature,

                "Spearman":
                    correlation,

                "Absolute_Spearman":
                    abs(correlation),
            })

    result = pd.DataFrame(
        windows
    )

    # --------------------------------------------------------
    # Summarize stability
    # --------------------------------------------------------

    summary = (
        result
        .groupby("Feature")
        .agg(

            Mean_IC=(
                "Spearman",
                "mean"
            ),

            Median_IC=(
                "Spearman",
                "median"
            ),

            IC_Std=(
                "Spearman",
                "std"
            ),

            Positive_Windows=(
                "Spearman",
                lambda x:
                (x > 0).sum()
            ),

            Windows=(
                "Spearman",
                "count"
            )
        )
        .reset_index()
    )

    summary[
        "Positive_Rate"
    ] = (
        summary[
            "Positive_Windows"
        ]
        /
        summary[
            "Windows"
        ]
    )

    summary[
        "Stable"
    ] = (

        (
            summary[
                "Positive_Rate"
            ] >= 0.75
        )

        &

        (
            summary[
                "Mean_IC"
            ].abs() >= 0.02
        )
    )

    return (
        result,
        summary.sort_values(
            "Mean_IC",
            key=lambda x:
            x.abs(),
            ascending=False
        )
    )


# ============================================================
# FEATURE GROUPS
# ============================================================

def feature_group(
    feature
):

    if feature.startswith(
        "Return_"
    ):

        return "Momentum"

    if feature in [
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "SMA20_vs_SMA50",
    ]:

        return "Trend"

    if feature in [
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Position",
    ]:

        return "Technical_Momentum"

    if feature in [
        "Volatility_20D",
        "ATR_Percent",
    ]:

        return "Volatility"

    if feature == "Volume_Ratio":

        return "Volume"

    if feature.startswith(
        "NIFTY_"
    ):

        return "NIFTY"

    if feature.startswith(
        "BANKNIFTY_"
    ):

        return "BANKNIFTY"

    if feature.startswith(
        "Trend_Strength"
    ):

        return "Relative_Strength"

    return "Other"


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_analysis(
    correlation_df,
    importance_df
):

    correlation_df = correlation_df.copy()

    importance_df = importance_df.copy()

    correlation_df[
        "Group"
    ] = correlation_df[
        "Feature"
    ].apply(
        feature_group
    )

    importance_df[
        "Group"
    ] = importance_df[
        "Feature"
    ].apply(
        feature_group
    )

    correlation_groups = (
        correlation_df
        .groupby("Group")
        .agg(

            Mean_Absolute_Spearman=(
                "Absolute_Spearman",
                "mean"
            ),

            Mean_Spearman=(
                "Spearman_Correlation",
                "mean"
            ),

            Features=(
                "Feature",
                "count"
            )
        )
        .reset_index()
    )

    importance_groups = (
        importance_df
        .groupby("Group")
        .agg(

            Mean_RF_Importance=(
                "RF_Importance",
                "mean"
            ),

            Total_RF_Importance=(
                "RF_Importance",
                "sum"
            ),

            Mean_Permutation=(
                "Permutation_Importance",
                "mean"
            )
        )
        .reset_index()
    )

    return correlation_groups.merge(
        importance_groups,
        on="Group"
    )


# ============================================================
# V24 COMPARISON
# ============================================================

def load_v24():

    if not V24_PREDICTIONS.exists():

        return None

    df = pd.read_csv(
        V24_PREDICTIONS
    )

    df = normalize_dates(
        df
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN STOCK TRADING AI")
    print("V26 - FEATURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    print(
        f"\nTotal rows: "
        f"{len(data):,}"
    )

    data, features = (
        prepare_data(data)
    )

    print(
        f"Features analyzed: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEATURE / TARGET CORRELATION")
    print("=" * 70)

    correlations = (
        correlation_analysis(
            data,
            features
        )
    )

    print(
        correlations[
            [
                "Feature",
                "Pearson_Correlation",
                "Spearman_Correlation",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RF importance
    # --------------------------------------------------------

    (
        importance,
        train,
        test,
        model
    ) = random_forest_importance(
        data,
        features
    )

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance[
            [
                "Feature",
                "RF_Importance",
                "Permutation_Importance",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Time stability
    # --------------------------------------------------------

    (
        stability_detail,
        stability_summary
    ) = time_stability(
        data,
        features
    )

    print("\n")
    print("=" * 70)
    print("FEATURE STABILITY")
    print("=" * 70)

    print(
        stability_summary[
            [
                "Feature",
                "Mean_IC",
                "Median_IC",
                "IC_Std",
                "Positive_Rate",
                "Stable",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = group_analysis(
        correlations,
        importance
    )

    print("\n")
    print("=" * 70)
    print("FEATURE GROUP ANALYSIS")
    print("=" * 70)

    print(
        groups
        .sort_values(
            "Total_RF_Importance",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Strong candidates
    # --------------------------------------------------------

    strong = (
        stability_summary[
            stability_summary[
                "Stable"
            ]
        ]
        .sort_values(
            "Mean_IC",
            key=lambda x:
            x.abs(),
            ascending=False
        )
    )

    print("\n")
    print("=" * 70)
    print("STABLE FEATURES")
    print("=" * 70)

    if strong.empty:

        print(
            "No features passed the "
            "stability criteria."
        )

    else:

        print(
            strong[
                [
                    "Feature",
                    "Mean_IC",
                    "Positive_Rate",
                    "IC_Std",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # V24 prediction distribution
    # --------------------------------------------------------

    v24 = load_v24()

    if v24 is not None:

        print("\n")
        print("=" * 70)
        print("V24 PREDICTION AUDIT")
        print("=" * 70)

        print(
            f"V24 predictions: "
            f"{len(v24):,}"
        )

        if (
            "Predicted_Excess_Return"
            in v24.columns
        ):

            print(
                v24[
                    "Predicted_Excess_Return"
                ].describe().to_string()
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    correlations_file = (
        OUTPUT_DIR
        / "v26_feature_correlations.csv"
    )

    importance_file = (
        OUTPUT_DIR
        / "v26_feature_importance.csv"
    )

    stability_file = (
        OUTPUT_DIR
        / "v26_feature_stability.csv"
    )

    stability_detail_file = (
        OUTPUT_DIR
        / "v26_feature_stability_detail.csv"
    )

    groups_file = (
        OUTPUT_DIR
        / "v26_feature_groups.csv"
    )

    correlations.to_csv(
        correlations_file,
        index=False
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    stability_summary.to_csv(
        stability_file,
        index=False
    )

    stability_detail.to_csv(
        stability_detail_file,
        index=False
    )

    groups.to_csv(
        groups_file,
        index=False
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        correlations_file.resolve()
    )

    print(
        importance_file.resolve()
    )

    print(
        stability_file.resolve()
    )

    print(
        stability_detail_file.resolve()
    )

    print(
        groups_file.resolve()
    )

    print("\n")
    print("=" * 70)
    print("V26 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()