from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "horizon_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
]


HORIZONS = [
    1,
    3,
    5,
    10,
    20,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_stock(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Find the date column automatically
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
    ]

    date_column = None

    for column in date_candidates:
        if column in df.columns:
            date_column = column
            break

    # --------------------------------------------------------
    # If no date column exists, check first column
    # --------------------------------------------------------

    if date_column is None:

        first_column = df.columns[0]

        try:
            parsed_dates = pd.to_datetime(
                df[first_column],
                errors="coerce"
            )

            if parsed_dates.notna().mean() > 0.90:

                date_column = first_column

        except Exception:
            pass

    # --------------------------------------------------------
    # Still no date column
    # --------------------------------------------------------

    if date_column is None:

        raise ValueError(
            f"Could not find date column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Standardize to lowercase 'date'
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df.dropna(
        subset=["date"],
        inplace=True
    )

    df.sort_values(
        "date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_features(df):

    excluded = {
        "date",
        "Future_Return",
        "Target",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        if column.startswith("News_"):
            continue

        if column == "Days_Since_News":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(column)

    return features


# ============================================================
# ANALYZE ONE HORIZON
# ============================================================

def analyze_horizon(
    df,
    features,
    horizon,
):

    data = df.copy()

    # --------------------------------------------------------
    # Future return
    # --------------------------------------------------------

    data["Future_Return"] = (
        data["Close"].shift(-horizon)
        / data["Close"]
        - 1
    ) * 100

    # --------------------------------------------------------
    # Remove rows without future return
    # --------------------------------------------------------

    data = data.dropna(
        subset=features + [
            "Future_Return"
        ]
    )

    if data.empty:

        return None

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlations = []

    for feature in features:

        correlation = (
            data[feature]
            .corr(
                data["Future_Return"]
            )
        )

        if pd.notna(correlation):

            correlations.append(
                {
                    "feature":
                        feature,

                    "correlation":
                        correlation,
                }
            )

    correlation_df = pd.DataFrame(
        correlations
    )

    if correlation_df.empty:

        strongest_positive = 0
        strongest_negative = 0

    else:

        strongest_positive = (
            correlation_df[
                "correlation"
            ].max()
        )

        strongest_negative = (
            correlation_df[
                "correlation"
            ].min()
        )

    # --------------------------------------------------------
    # Absolute correlation
    # --------------------------------------------------------

    if not correlation_df.empty:

        strongest_absolute = (
            correlation_df
            .assign(
                abs_corr=lambda x:
                    x["correlation"].abs()
            )
            .sort_values(
                "abs_corr",
                ascending=False
            )
            .iloc[0]
        )

        strongest_feature = (
            strongest_absolute[
                "feature"
            ]
        )

        strongest_feature_corr = (
            strongest_absolute[
                "correlation"
            ]
        )

    else:

        strongest_feature = ""

        strongest_feature_corr = 0

    # --------------------------------------------------------
    # Overall return statistics
    # --------------------------------------------------------

    average_return = (
        data[
            "Future_Return"
        ].mean()
    )

    median_return = (
        data[
            "Future_Return"
        ].median()
    )

    win_rate = (
        data[
            "Future_Return"
        ]
        .gt(0)
        .mean()
        * 100
    )

    return_std = (
        data[
            "Future_Return"
        ].std()
    )

    # --------------------------------------------------------
    # Simple momentum signal
    #
    # Test whether positive recent momentum predicts
    # future returns.
    # --------------------------------------------------------

    momentum_results = []

    momentum_columns = [
        "Return_1D",
        "Return_5D",
        "Return_10D",
        "Return_20D",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "SMA20_vs_SMA50",
    ]

    for column in momentum_columns:

        if column not in data.columns:

            continue

        positive = data[
            data[column] > 0
        ]

        negative = data[
            data[column] <= 0
        ]

        if (
            not positive.empty
            and not negative.empty
        ):

            positive_return = (
                positive[
                    "Future_Return"
                ].mean()
            )

            negative_return = (
                negative[
                    "Future_Return"
                ].mean()
            )

            momentum_results.append(
                {
                    "horizon":
                        horizon,

                    "feature":
                        column,

                    "positive_condition_return":
                        positive_return,

                    "negative_condition_return":
                        negative_return,

                    "difference":
                        (
                            positive_return
                            - negative_return
                        ),
                }
            )

    return {
        "horizon":
            horizon,

        "observations":
            len(data),

        "average_return_pct":
            average_return,

        "median_return_pct":
            median_return,

        "win_rate_pct":
            win_rate,

        "return_std_pct":
            return_std,

        "strongest_positive_correlation":
            strongest_positive,

        "strongest_negative_correlation":
            strongest_negative,

        "strongest_feature":
            strongest_feature,

        "strongest_feature_correlation":
            strongest_feature_corr,

        "momentum_results":
            momentum_results,

        "correlations":
            correlation_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "PREDICTION HORIZON ANALYSIS"
    )
    print("=" * 80)

    all_summary = []

    all_momentum = []

    all_correlations = []

    # --------------------------------------------------------
    # Process stocks
    # --------------------------------------------------------

    for symbol in STOCKS:

        print(
            f"\nProcessing {symbol}..."
        )

        try:

            df = load_stock(
                symbol
            )

            features = get_features(
                df
            )

            for horizon in HORIZONS:

                result = analyze_horizon(
                    df,
                    features,
                    horizon
                )

                if result is None:

                    continue

                all_summary.append(
                    {
                        "symbol":
                            symbol,

                        "horizon":
                            horizon,

                        "observations":
                            result[
                                "observations"
                            ],

                        "average_return_pct":
                            result[
                                "average_return_pct"
                            ],

                        "median_return_pct":
                            result[
                                "median_return_pct"
                            ],

                        "win_rate_pct":
                            result[
                                "win_rate_pct"
                            ],

                        "return_std_pct":
                            result[
                                "return_std_pct"
                            ],

                        "strongest_positive_correlation":
                            result[
                                "strongest_positive_correlation"
                            ],

                        "strongest_negative_correlation":
                            result[
                                "strongest_negative_correlation"
                            ],

                        "strongest_feature":
                            result[
                                "strongest_feature"
                            ],

                        "strongest_feature_correlation":
                            result[
                                "strongest_feature_correlation"
                            ],
                    }
                )

                for item in result[
                    "momentum_results"
                ]:

                    item["symbol"] = symbol

                    all_momentum.append(
                        item
                    )

                correlations = result[
                    "correlations"
                ].copy()

                if not correlations.empty:

                    correlations[
                        "symbol"
                    ] = symbol

                    correlations[
                        "horizon"
                    ] = horizon

                    all_correlations.append(
                        correlations
                    )

            print(
                "OK"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    summary = pd.DataFrame(
        all_summary
    )

    momentum = pd.DataFrame(
        all_momentum
    )

    if all_correlations:

        correlations = pd.concat(
        all_correlations,
        ignore_index=True
    )

    else:

        correlations = pd.DataFrame(
        columns=[
            "symbol",
            "horizon",
            "feature",
            "correlation",
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / "horizon_summary.csv"
    )

    momentum_file = (
        OUTPUT_DIR
        / "momentum_analysis.csv"
    )

    correlations_file = (
        OUTPUT_DIR
        / "feature_correlations.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    momentum.to_csv(
        momentum_file,
        index=False
    )

    correlations.to_csv(
        correlations_file,
        index=False
    )

    # ========================================================
    # OVERALL HORIZON SUMMARY
    # ========================================================

    overall = (
        summary
        .groupby("horizon")
        .agg(
            stocks=(
                "symbol",
                "nunique"
            ),

            observations=(
                "observations",
                "sum"
            ),

            average_return_pct=(
                "average_return_pct",
                "mean"
            ),

            median_return_pct=(
                "median_return_pct",
                "mean"
            ),

            average_win_rate_pct=(
                "win_rate_pct",
                "mean"
            ),

            average_return_std_pct=(
                "return_std_pct",
                "mean"
            ),

            average_strongest_correlation=(
                "strongest_feature_correlation",
                "mean"
            ),
        )
        .reset_index()
    )

    overall_file = (
        OUTPUT_DIR
        / "overall_horizon_summary.csv"
    )

    overall.to_csv(
        overall_file,
        index=False
    )

    # --------------------------------------------------------
    # Print overall
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "OVERALL HORIZON RESULTS"
    )
    print("=" * 80)

    print(
        overall.to_string(
            index=False
        )
    )

    # ========================================================
    # BEST HORIZON BY CORRELATION
    # ========================================================

    best_horizon = (
        overall.sort_values(
            "average_strongest_correlation",
            key=lambda x:
                x.abs(),
            ascending=False
        )
        .iloc[0]
    )

    print("\n")
    print("=" * 80)
    print(
        "STRONGEST HORIZON"
    )
    print("=" * 80)

    print(
        f"\nHorizon: "
        f"{int(best_horizon['horizon'])} days"
    )

    print(
        f"Average strongest feature correlation: "
        f"{best_horizon['average_strongest_correlation']:.4f}"
    )

    # ========================================================
    # TOP FEATURES BY HORIZON
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "TOP FEATURES BY HORIZON"
    )
    print("=" * 80)

    for horizon in HORIZONS:

        subset = correlations[
            correlations[
                "horizon"
            ] == horizon
        ].copy()

        if subset.empty:

            continue

        subset[
            "abs_correlation"
        ] = subset[
            "correlation"
        ].abs()

        top = (
            subset
            .sort_values(
                "abs_correlation",
                ascending=False
            )
            .head(10)
        )

        print(
            f"\n--- {horizon}-DAY HORIZON ---"
        )

        print(
            top[
                [
                    "symbol",
                    "feature",
                    "correlation",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # BEST MOMENTUM CONDITIONS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "BEST MOMENTUM CONDITIONS"
    )
    print("=" * 80)

    if not momentum.empty:

        best_momentum = (
            momentum
            .sort_values(
                "difference",
                ascending=False
            )
            .head(15)
        )

        print(
            best_momentum[
                [
                    "symbol",
                    "horizon",
                    "feature",
                    "positive_condition_return",
                    "negative_condition_return",
                    "difference",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "HORIZON ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()