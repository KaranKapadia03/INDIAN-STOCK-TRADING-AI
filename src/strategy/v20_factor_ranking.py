from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/processed")

OUTPUT_DIR = Path(
    "data/models/v20_factor_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZON = 20

TRAIN_RATIO = 0.60


# ============================================================
# FACTORS
# ============================================================

FACTOR_COLUMNS = [
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "SMA20_vs_SMA50",
    "RSI_14",
    "MACD_Histogram",
    "BB_Position",
    "Volatility_20D",
    "ATR_Percent",
    "Volume_Ratio",
    "Trend_Strength_vs_NIFTY",
    "Trend_Strength_vs_BANKNIFTY",
]


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_feature_files():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    return [
        f for f in files
        if "_features_features" not in f.name
    ]


def extract_symbol(
    path
):

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
# DATE NORMALIZATION
# ============================================================

def normalize_dates(
    df
):

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
# LOAD DATA
# ============================================================

def load_data():

    all_data = []

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

        df["Symbol"] = symbol

        # ----------------------------------------------------
        # Future return
        # ----------------------------------------------------

        df[
            "Future_Return_20D"
        ] = (

            df["Close"]
            .shift(-HORIZON)
            / df["Close"]
            - 1
        ) * 100

        available = [
            x
            for x in FACTOR_COLUMNS
            if x in df.columns
        ]

        required = [
            "Date",
            "Symbol",
            "Close",
            "Future_Return_20D",
        ]

        required += available

        df = df[
            required
        ].copy()

        all_data.append(
            df
        )

        print(
            f"{symbol:<18}"
            f"{len(df):>5} rows"
        )

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    return data


# ============================================================
# CROSS-SECTIONAL Z-SCORE
# ============================================================

def cross_sectional_zscore(
    data,
    column
):

    mean = (
        data
        .groupby("Date")[column]
        .transform("mean")
    )

    std = (
        data
        .groupby("Date")[column]
        .transform("std")
    )

    z = (
        data[column]
        - mean
    ) / std.replace(
        0,
        np.nan
    )

    return z.clip(
        -3,
        3
    )


# ============================================================
# FACTOR SCORE
# ============================================================

def build_factor_score(
    data
):

    data = data.copy()

    factor_scores = []

    # --------------------------------------------------------
    # Factors where HIGHER is better
    # --------------------------------------------------------

    positive_factors = [

        "Return_5D",

        "Return_10D",

        "Return_20D",

        "Price_vs_SMA20",

        "Price_vs_SMA50",

        "SMA20_vs_SMA50",

        "MACD_Histogram",

        "Trend_Strength_vs_NIFTY",

        "Trend_Strength_vs_BANKNIFTY",
    ]

    # --------------------------------------------------------
    # RSI
    #
    # Very low RSI can indicate weakness.
    # Extremely high RSI can indicate overextension.
    #
    # We therefore reward RSI around 55-65.
    # --------------------------------------------------------

    if "RSI_14" in data.columns:

        rsi_distance = (
            abs(
                data["RSI_14"]
                - 60
            )
        )

        data[
            "RSI_Factor"
        ] = -rsi_distance

        positive_factors.append(
            "RSI_Factor"
        )

    # --------------------------------------------------------
    # Bollinger position
    #
    # Slightly positive positioning is preferred,
    # but extreme values are avoided.
    # --------------------------------------------------------

    if "BB_Position" in data.columns:

        bb_distance = (
            abs(
                data["BB_Position"]
                - 0.60
            )
        )

        data[
            "BB_Factor"
        ] = -bb_distance

        positive_factors.append(
            "BB_Factor"
        )

    # --------------------------------------------------------
    # Volume
    #
    # Higher-than-normal volume gets a modest positive score.
    # --------------------------------------------------------

    if "Volume_Ratio" in data.columns:

        volume_factor = (
            np.log1p(
                data[
                    "Volume_Ratio"
                ]
                .clip(
                    0,
                    5
                )
            )
        )

        data[
            "Volume_Factor"
        ] = volume_factor

        positive_factors.append(
            "Volume_Factor"
        )

    # --------------------------------------------------------
    # Create z-scores
    # --------------------------------------------------------

    for factor in positive_factors:

        if factor not in data.columns:
            continue

        z = cross_sectional_zscore(
            data,
            factor
        )

        factor_scores.append(
            z.rename(
                factor
            )
        )

    factor_df = pd.concat(
        factor_scores,
        axis=1
    )

    # --------------------------------------------------------
    # Equal-weight factor score
    # --------------------------------------------------------

    data[
        "Factor_Score"
    ] = factor_df.mean(
        axis=1
    )

    return data


# ============================================================
# DAILY RANKING
# ============================================================

def rank_stocks(
    data
):

    data = data.copy()

    data[
        "Factor_Rank"
    ] = (
        data
        .groupby("Date")[
            "Factor_Score"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )

    data[
        "Stock_Count"
    ] = (
        data
        .groupby("Date")[
            "Symbol"
        ]
        .transform("count")
    )

    data[
        "Top_5"
    ] = (
        data[
            "Factor_Rank"
        ] <= 5
    )

    data[
        "Bottom_5"
    ] = (
        data[
            "Factor_Rank"
        ]
        > (
            data[
                "Stock_Count"
            ]
            - 5
        )
    )

    return data


# ============================================================
# RANK IC
# ============================================================

def daily_rank_ic(
    data
):

    values = []

    for date, group in (
        data.groupby("Date")
    ):

        if len(group) < 5:
            continue

        correlation = (
            group[
                "Factor_Score"
            ]
            .corr(
                group[
                    "Future_Return_20D"
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
# PERFORMANCE
# ============================================================

def evaluate(
    data,
    label
):

    top = data[
        data["Top_5"]
    ]

    bottom = data[
        data["Bottom_5"]
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

    top_win_rate = (
        top[
            "Future_Return_20D"
        ] > 0
    ).mean()

    ic_mean, ic_median = (
        daily_rank_ic(
            data
        )
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        label
    )

    print(
        "=" * 70
    )

    print(
        f"Top-5 Return: "
        f"{top_return:.3f}%"
    )

    print(
        f"Bottom-5 Return: "
        f"{bottom_return:.3f}%"
    )

    print(
        f"Top-Bottom Spread: "
        f"{spread:.3f}%"
    )

    print(
        f"Top-5 Win Rate: "
        f"{top_win_rate * 100:.2f}%"
    )

    print(
        f"Mean Daily Rank IC: "
        f"{ic_mean:.4f}"
    )

    print(
        f"Median Daily Rank IC: "
        f"{ic_median:.4f}"
    )

    return {

        "Period":
            label,

        "Top5_Return":
            top_return,

        "Bottom5_Return":
            bottom_return,

        "Spread":
            spread,

        "Top5_Win_Rate":
            top_win_rate,

        "Mean_Rank_IC":
            ic_mean,

        "Median_Rank_IC":
            ic_median,
    }


# ============================================================
# FACTOR ATTRIBUTION
# ============================================================

def factor_analysis(
    data,
    factors
):

    rows = []

    for factor in factors:

        if factor not in data.columns:
            continue

        ic_values = []

        for _, group in (
            data.groupby("Date")
        ):

            if len(group) < 5:
                continue

            ic = (
                group[
                    factor
                ]
                .corr(
                    group[
                        "Future_Return_20D"
                    ],
                    method="spearman"
                )
            )

            if pd.notna(ic):
                ic_values.append(ic)

        if ic_values:

            rows.append({

                "Factor":
                    factor,

                "Mean_IC":
                    np.mean(
                        ic_values
                    ),

                "Median_IC":
                    np.median(
                        ic_values
                    ),

                "Positive_IC_Rate":
                    np.mean(
                        np.array(
                            ic_values
                        ) > 0
                    ),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Mean_IC",
            ascending=False
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
        "V20 - FACTOR RANKING BASELINE"
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

    factor_columns = [
        f
        for f in FACTOR_COLUMNS
        if f in data.columns
    ]

    # Fill missing values
    # cross-sectionally by date

    for factor in factor_columns:

        data[
            factor
        ] = (
            data
            .groupby("Date")[
                factor
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
            "Future_Return_20D"
        ]
    )

    # ========================================================
    # FACTOR SCORE
    # ========================================================

    data = (
        build_factor_score(
            data
        )
    )

    data = data.dropna(
        subset=[
            "Factor_Score"
        ]
    )

    data = rank_stocks(
        data
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
        "OUT-OF-SAMPLE SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Development: "
        f"{train['Date'].min().date()} "
        f"-> "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Test: "
        f"{test['Date'].min().date()} "
        f"-> "
        f"{test['Date'].max().date()}"
    )

    # ========================================================
    # EVALUATE FULL PERIOD
    # ========================================================

    full_result = evaluate(
        data,
        "FULL DATA"
    )

    # ========================================================
    # DEVELOPMENT
    # ========================================================

    train_result = evaluate(
        train,
        "DEVELOPMENT PERIOD"
    )

    # ========================================================
    # OUT OF SAMPLE
    # ========================================================

    test_result = evaluate(
        test,
        "OUT-OF-SAMPLE TEST"
    )

    # ========================================================
    # FACTOR ANALYSIS
    # ========================================================

    print("\n")
    print(
        "=" * 70
    )

    print(
        "FACTOR ATTRIBUTION"
    )

    print(
        "=" * 70
    )

    factor_results = factor_analysis(
        test,
        factor_columns
    )

    if not factor_results.empty:

        print(
            factor_results.to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    data_file = (
        OUTPUT_DIR
        / "v20_factor_scores.csv"
    )

    results_file = (
        OUTPUT_DIR
        / "v20_results.csv"
    )

    factor_file = (
        OUTPUT_DIR
        / "v20_factor_analysis.csv"
    )

    data.to_csv(
        data_file,
        index=False
    )

    pd.DataFrame(
        [
            full_result,
            train_result,
            test_result,
        ]
    ).to_csv(
        results_file,
        index=False
    )

    factor_results.to_csv(
        factor_file,
        index=False
    )

    # ========================================================
    # CURRENT RANKING
    # ========================================================

    latest_date = (
        data["Date"].max()
    )

    latest = (
        data[
            data["Date"]
            == latest_date
        ]
        .sort_values(
            "Factor_Score",
            ascending=False
        )
    )

    latest[
        [
            "Symbol",
            "Factor_Score",
            "Factor_Rank",
            "Return_20D",
            "RSI_14",
            "Volatility_20D",
        ]
    ].to_csv(
        OUTPUT_DIR
        / "current_ranking.csv",
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
        latest[
            [
                "Symbol",
                "Factor_Score",
                "Factor_Rank",
                "Return_20D",
                "RSI_14",
                "Volatility_20D",
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
        data_file.resolve()
    )

    print(
        results_file.resolve()
    )

    print(
        factor_file.resolve()
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "V20 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()