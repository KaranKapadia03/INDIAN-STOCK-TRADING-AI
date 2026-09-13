from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

OUTPUT_DIR = Path(
    "data/models/v21_data_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = [1, 5, 10, 20]


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
# LOAD
# ============================================================

def load_data():

    datasets = {}

    for file in get_feature_files():

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
            .reset_index(drop=True)
        )

        datasets[
            symbol
        ] = df

    return datasets


# ============================================================
# BASIC DATA AUDIT
# ============================================================

def audit_basic_data(
    datasets
):

    rows = []

    for symbol, df in datasets.items():

        duplicate_dates = (
            df["Date"]
            .duplicated()
            .sum()
        )

        missing_dates = (
            df["Date"]
            .isna()
            .sum()
        )

        missing_close = (
            df["Close"]
            .isna()
            .sum()
        )

        invalid_prices = (
            (
                df["Close"]
                <= 0
            )
            .sum()
        )

        rows.append({

            "Symbol":
                symbol,

            "Rows":
                len(df),

            "Start":
                df["Date"].min(),

            "End":
                df["Date"].max(),

            "Duplicate_Dates":
                duplicate_dates,

            "Missing_Dates":
                missing_dates,

            "Missing_Close":
                missing_close,

            "Invalid_Close":
                invalid_prices,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# OHLC AUDIT
# ============================================================

def audit_ohlc(
    datasets
):

    rows = []

    for symbol, df in datasets.items():

        invalid_high = (
            df["High"]
            < df[
                [
                    "Open",
                    "Close",
                    "Low"
                ]
            ].max(axis=1)
        ).sum()

        invalid_low = (
            df["Low"]
            > df[
                [
                    "Open",
                    "Close",
                    "High"
                ]
            ].min(axis=1)
        ).sum()

        negative_volume = (
            df["Volume"]
            < 0
        ).sum()

        rows.append({

            "Symbol":
                symbol,

            "Invalid_High":
                invalid_high,

            "Invalid_Low":
                invalid_low,

            "Negative_Volume":
                negative_volume,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET AUDIT
# ============================================================

def create_targets(
    df
):

    df = df.copy()

    for horizon in HORIZONS:

        df[
            f"Future_Return_{horizon}D"
        ] = (

            df["Close"]
            .shift(-horizon)
            / df["Close"]
            - 1
        ) * 100

    return df


def audit_targets(
    datasets
):

    rows = []

    for symbol, original in datasets.items():

        df = create_targets(
            original
        )

        for horizon in HORIZONS:

            target = df[
                f"Future_Return_{horizon}D"
            ].dropna()

            rows.append({

                "Symbol":
                    symbol,

                "Horizon":
                    horizon,

                "Count":
                    len(target),

                "Mean":
                    target.mean(),

                "Median":
                    target.median(),

                "Std":
                    target.std(),

                "Min":
                    target.min(),

                "Max":
                    target.max(),

                "Positive_Rate":
                    (
                        target > 0
                    ).mean(),

                "Greater_5_Rate":
                    (
                        target >= 5
                    ).mean(),

                "Less_Minus5_Rate":
                    (
                        target <= -5
                    ).mean(),

            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CROSS-SECTIONAL AUDIT
# ============================================================

def cross_sectional_audit(
    datasets
):

    all_data = []

    for symbol, original in datasets.items():

        df = create_targets(
            original
        )

        df = df[
            [
                "Date",
                "Close",
                "Future_Return_20D"
            ]
        ].copy()

        df["Symbol"] = symbol

        all_data.append(
            df
        )

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Number of stocks available each day
    # --------------------------------------------------------

    coverage = (
        data
        .groupby("Date")
        .agg(
            Stocks=(
                "Symbol",
                "nunique"
            ),

            Rows=(
                "Symbol",
                "count"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Cross-sectional target
    # --------------------------------------------------------

    data[
        "Daily_Mean_Return"
    ] = (
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
        - data[
            "Daily_Mean_Return"
        ]
    )

    return (
        data,
        coverage
    )


# ============================================================
# FEATURE LOOK-AHEAD AUDIT
# ============================================================

def audit_features(
    datasets
):

    rows = []

    for symbol, df in datasets.items():

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numeric_columns:

            if column in [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]:

                continue

            # Correlation with NEXT day's return.
            #
            # This isn't itself leakage.
            # It simply helps identify suspiciously
            # predictive features.

            next_return = (
                df["Close"]
                .shift(-1)
                / df["Close"]
                - 1
            )

            valid = (
                pd.concat(
                    [
                        df[column],
                        next_return
                    ],
                    axis=1
                )
                .dropna()
            )

            if len(valid) < 50:

                continue

            correlation = (
                valid.iloc[:, 0]
                .corr(
                    valid.iloc[:, 1]
                )
            )

            rows.append({

                "Symbol":
                    symbol,

                "Feature":
                    column,

                "Next_Day_Correlation":
                    correlation,

            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET ALIGNMENT TEST
# ============================================================

def target_alignment_test(
    datasets
):

    rows = []

    for symbol, original in datasets.items():

        df = original.copy()

        # Manual target

        manual = (
            df["Close"]
            .shift(-20)
            / df["Close"]
            - 1
        ) * 100

        # Alternative formulation

        alternative = (
            (
                df["Close"]
                .shift(-20)
                - df["Close"]
            )
            / df["Close"]
        ) * 100

        difference = (
            manual
            - alternative
        ).abs()

        rows.append({

            "Symbol":
                symbol,

            "Max_Difference":
                difference.max(),

            "Mean_Difference":
                difference.mean(),

            "Valid_Comparisons":
                difference.notna().sum(),

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# NEXT-DAY EXECUTION TEST
# ============================================================

def execution_comparison(
    datasets
):

    rows = []

    for symbol, df in datasets.items():

        close_return = (
            df["Close"]
            .shift(-20)
            / df["Close"]
            - 1
        ) * 100

        next_open_return = (
            df["Close"]
            .shift(-20)
            / df["Open"].shift(-1)
            - 1
        ) * 100

        valid = pd.concat(
            [
                close_return,
                next_open_return
            ],
            axis=1
        ).dropna()

        rows.append({

            "Symbol":
                symbol,

            "Close_To_Close_Mean":
                valid.iloc[:, 0].mean(),

            "Next_Open_To_Close_Mean":
                valid.iloc[:, 1].mean(),

            "Close_To_Close_Median":
                valid.iloc[:, 0].median(),

            "Next_Open_To_Close_Median":
                valid.iloc[:, 1].median(),

        })

    return pd.DataFrame(
        rows
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
        "V21 - DATA & TARGET AUDIT"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    datasets = load_data()

    print(
        f"\nStocks loaded: "
        f"{len(datasets)}"
    )

    # ========================================================
    # BASIC
    # ========================================================

    basic = audit_basic_data(
        datasets
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "BASIC DATA AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        basic.to_string(
            index=False
        )
    )

    # ========================================================
    # OHLC
    # ========================================================

    ohlc = audit_ohlc(
        datasets
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "OHLC AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        ohlc.to_string(
            index=False
        )
    )

    # ========================================================
    # TARGET
    # ========================================================

    targets = audit_targets(
        datasets
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "TARGET DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    target_summary = (
        targets
        .groupby("Horizon")
        .agg(
            Count=(
                "Count",
                "sum"
            ),

            Mean=(
                "Mean",
                "mean"
            ),

            Median=(
                "Median",
                "mean"
            ),

            Std=(
                "Std",
                "mean"
            ),

            Positive_Rate=(
                "Positive_Rate",
                "mean"
            ),

            Greater_5_Rate=(
                "Greater_5_Rate",
                "mean"
            ),

            Less_Minus5_Rate=(
                "Less_Minus5_Rate",
                "mean"
            ),
        )
    )

    print(
        target_summary.to_string()
    )

    # ========================================================
    # CROSS SECTION
    # ========================================================

    cross_data, coverage = (
        cross_sectional_audit(
            datasets
        )
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "CROSS-SECTIONAL COVERAGE"
    )

    print(
        "=" * 70
    )

    print(
        f"Average stocks/day: "
        f"{coverage['Stocks'].mean():.2f}"
    )

    print(
        f"Minimum stocks/day: "
        f"{coverage['Stocks'].min()}"
    )

    print(
        f"Maximum stocks/day: "
        f"{coverage['Stocks'].max()}"
    )

    print(
        f"Dates with all 20 stocks: "
        f"{(
            coverage['Stocks'] == 20
        ).sum():,}"
    )

    print(
        f"Total dates: "
        f"{len(coverage):,}"
    )

    # ========================================================
    # RELATIVE TARGET
    # ========================================================

    relative = (
        cross_data[
            "Relative_Return_20D"
        ]
        .dropna()
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "RELATIVE TARGET"
    )

    print(
        "=" * 70
    )

    print(
        f"Mean: "
        f"{relative.mean():.6f}%"
    )

    print(
        f"Median: "
        f"{relative.median():.6f}%"
    )

    print(
        f"Std: "
        f"{relative.std():.4f}%"
    )

    # ========================================================
    # TARGET ALIGNMENT
    # ========================================================

    alignment = (
        target_alignment_test(
            datasets
        )
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "TARGET ALIGNMENT TEST"
    )

    print(
        "=" * 70
    )

    print(
        alignment.to_string(
            index=False
        )
    )

    # ========================================================
    # EXECUTION
    # ========================================================

    execution = (
        execution_comparison(
            datasets
        )
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "EXECUTION COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        execution.to_string(
            index=False
        )
    )

    # ========================================================
    # FEATURE AUDIT
    # ========================================================

    print("\n")
    print(
        "=" * 70
    )

    print(
        "FEATURE AUDIT"
    )

    print(
        "=" * 70
    )

    feature_audit = audit_features(
        datasets
    )

    suspicious = (
        feature_audit
        .sort_values(
            "Next_Day_Correlation",
            key=lambda x:
            x.abs(),
            ascending=False
        )
        .head(30)
    )

    print(
        suspicious.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    basic.to_csv(
        OUTPUT_DIR
        / "basic_audit.csv",
        index=False
    )

    ohlc.to_csv(
        OUTPUT_DIR
        / "ohlc_audit.csv",
        index=False
    )

    targets.to_csv(
        OUTPUT_DIR
        / "target_audit.csv",
        index=False
    )

    coverage.to_csv(
        OUTPUT_DIR
        / "cross_sectional_coverage.csv",
        index=False
    )

    alignment.to_csv(
        OUTPUT_DIR
        / "target_alignment.csv",
        index=False
    )

    execution.to_csv(
        OUTPUT_DIR
        / "execution_comparison.csv",
        index=False
    )

    feature_audit.to_csv(
        OUTPUT_DIR
        / "feature_audit.csv",
        index=False
    )

    cross_data.to_csv(
        OUTPUT_DIR
        / "cross_sectional_data.csv",
        index=False
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    problems = 0

    problems += int(
        basic[
            "Duplicate_Dates"
        ].sum()
        > 0
    )

    problems += int(
        basic[
            "Missing_Close"
        ].sum()
        > 0
    )

    problems += int(
        ohlc[
            "Invalid_High"
        ].sum()
        > 0
    )

    problems += int(
        ohlc[
            "Invalid_Low"
        ].sum()
        > 0
    )

    print("\n")
    print(
        "=" * 70
    )

    if problems == 0:

        print(
            "DATA INTEGRITY: PASS"
        )

    else:

        print(
            f"DATA INTEGRITY: "
            f"{problems} ISSUE(S) FOUND"
        )

    print(
        "=" * 70
    )

    print(
        "\nSaved audit files to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print("\n")
    print(
        "V21 COMPLETE"
    )


if __name__ == "__main__":
    main()