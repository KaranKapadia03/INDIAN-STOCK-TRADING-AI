
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path(
    "data/processed"
)

PREDICTION_FILE = Path(
    "data/models/v29_realistic_backtest/"
    "v29_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/models/v32_prediction_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HORIZONS = [
    1,
    5,
    10,
    20,
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(df):

    df = df.copy()

    if "Date" in df.columns:

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
# SYMBOL EXTRACTION
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
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    print(
        f"Feature files found: "
        f"{len(files)}"
    )

    datasets = []

    for file in files:

        symbol = extract_symbol(
            file
        )

        df = pd.read_csv(
            file
        )

        required = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volatility_20D",
            "Volume_Ratio",
            "NIFTY_Return_5D",
            "NIFTY_Return_20D",
            "BANKNIFTY_Return_5D",
            "BANKNIFTY_Return_20D",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            print(
                f"Skipping {symbol}: "
                f"{missing}"
            )

            continue

        df = normalize_dates(
            df
        )

        df = (
            df
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        df["Symbol"] = symbol

        # ----------------------------------------------------
        # Future returns at every horizon
        # ----------------------------------------------------

        for horizon in HORIZONS:

            df[
                f"Future_Return_{horizon}D"
            ] = (

                df["Close"].shift(
                    -horizon
                )
                / df["Close"]
                - 1

            ) * 100

        # ----------------------------------------------------
        # Next-open returns
        # ----------------------------------------------------

        df[
            "Next_Open_Return"
        ] = (

            df["Open"].shift(-1)
            / df["Close"]
            - 1

        ) * 100

        # ----------------------------------------------------
        # Next-open to future close
        # ----------------------------------------------------

        for horizon in HORIZONS:

            if horizon == 1:

                future_open = (
                    df["Open"].shift(-1)
                )

                future_close = (
                    df["Close"].shift(-1)
                )

            else:

                future_open = (
                    df["Open"].shift(-1)
                )

                future_close = (
                    df["Close"].shift(
                        -horizon
                    )
                )

            df[
                f"Open_to_Close_{horizon}D"
            ] = (

                future_close
                / future_open
                - 1

            ) * 100

        datasets.append(
            df
        )

    if not datasets:

        raise RuntimeError(
            "No usable market datasets."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    data = data.loc[
        :,
        ~data.columns.duplicated()
    ]

    return (
        data
        .sort_values(
            [
                "Date",
                "Symbol",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n"
            f"{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions = normalize_dates(
        predictions
    )

    required = [
        "Date",
        "Symbol",
        "Prediction",
    ]

    missing = [
        column
        for column in required
        if column not in predictions.columns
    ]

    if missing:

        raise RuntimeError(
            f"Prediction file missing: "
            f"{missing}"
        )

    return (
        predictions
        .sort_values(
            [
                "Date",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# MERGE PREDICTIONS + FUTURE RETURNS
# ============================================================

def merge_data(
    predictions,
    market_data
):

    future_columns = [

        "Date",
        "Symbol",

        "Close",

        "NIFTY_Return_5D",
        "NIFTY_Return_20D",

        "BANKNIFTY_Return_5D",
        "BANKNIFTY_Return_20D",

    ]

    for horizon in HORIZONS:

        future_columns.append(
            f"Future_Return_{horizon}D"
        )

        future_columns.append(
            f"Open_to_Close_{horizon}D"
        )

    merged = predictions.merge(

        market_data[
            future_columns
        ],

        on=[
            "Date",
            "Symbol",
        ],

        how="left",

    )

    merged = merged.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    return merged


# ============================================================
# OVERALL CORRELATION
# ============================================================

def overall_correlation(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "1. OVERALL PREDICTION CORRELATION"
    )
    print("=" * 70)

    rows = []

    for horizon in HORIZONS:

        target = (
            f"Future_Return_{horizon}D"
        )

        df = data[
            [
                "Prediction",
                target,
            ]
        ].dropna()

        if df.empty:

            continue

        pearson = df[
            "Prediction"
        ].corr(
            df[target],
            method="pearson"
        )

        spearman = df[
            "Prediction"
        ].corr(
            df[target],
            method="spearman"
        )

        rows.append({

            "Horizon":
                horizon,

            "Observations":
                len(df),

            "Pearson":
                pearson,

            "Spearman":
                spearman,

        })

        print(
            f"{horizon}D | "
            f"Pearson: {pearson:.4f} | "
            f"Spearman: {spearman:.4f}"
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DAILY RANK IC
# ============================================================

def rank_ic_analysis(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "2. DAILY CROSS-SECTIONAL RANK IC"
    )
    print("=" * 70)

    all_rows = []

    summary_rows = []

    for horizon in HORIZONS:

        target = (
            f"Future_Return_{horizon}D"
        )

        daily_rows = []

        for date, group in (
            data.groupby("Date")
        ):

            group = group[
                [
                    "Symbol",
                    "Prediction",
                    target,
                ]
            ].dropna()

            if len(group) < 3:

                continue

            if (
                group[
                    "Prediction"
                ].nunique()
                < 2
            ):

                continue

            if (
                group[target].nunique()
                < 2
            ):

                continue

            ic = group[
                "Prediction"
            ].corr(
                group[target],
                method="spearman"
            )

            if pd.notna(ic):

                daily_rows.append({

                    "Date":
                        date,

                    "Horizon":
                        horizon,

                    "Rank_IC":
                        ic,

                })

        ic_df = pd.DataFrame(
            daily_rows
        )

        if ic_df.empty:

            continue

        mean_ic = (
            ic_df["Rank_IC"].mean()
        )

        median_ic = (
            ic_df["Rank_IC"].median()
        )

        positive_rate = (
            (
                ic_df["Rank_IC"]
                > 0
            ).mean()
        )

        print(
            f"{horizon}D | "
            f"Mean IC: {mean_ic:.4f} | "
            f"Median IC: {median_ic:.4f} | "
            f"Positive IC: "
            f"{positive_rate * 100:.2f}%"
        )

        summary_rows.append({

            "Horizon":
                horizon,

            "Days":
                len(ic_df),

            "Mean_Rank_IC":
                mean_ic,

            "Median_Rank_IC":
                median_ic,

            "Positive_IC_Rate":
                positive_rate,

        })

        all_rows.append(
            ic_df
        )

    if all_rows:

        daily_result = pd.concat(
            all_rows,
            ignore_index=True
        )

    else:

        daily_result = pd.DataFrame()

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        summary,
        daily_result
    )


# ============================================================
# TOP-N PORTFOLIO ANALYSIS
# ============================================================

def top_n_analysis(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "3. TOP-N RANKING ANALYSIS"
    )
    print("=" * 70)

    rows = []

    for horizon in HORIZONS:

        target = (
            f"Future_Return_{horizon}D"
        )

        for top_n in [
            1,
            3,
            5,
        ]:

            daily_returns = []

            bottom_returns = []

            for date, group in (
                data.groupby("Date")
            ):

                group = group[
                    [
                        "Prediction",
                        target,
                    ]
                ].dropna()

                if len(group) < top_n:

                    continue

                group = (
                    group
                    .sort_values(
                        "Prediction",
                        ascending=False
                    )
                )

                top = group.head(
                    top_n
                )

                bottom = group.tail(
                    top_n
                )

                daily_returns.append(
                    top[target].mean()
                )

                bottom_returns.append(
                    bottom[target].mean()
                )

            if not daily_returns:

                continue

            top_mean = np.mean(
                daily_returns
            )

            bottom_mean = np.mean(
                bottom_returns
            )

            spread = (
                top_mean
                - bottom_mean
            )

            top_win = (
                np.mean(
                    np.array(
                        daily_returns
                    ) > 0
                )
            )

            rows.append({

                "Horizon":
                    horizon,

                "Top_N":
                    top_n,

                "Top_Mean_Return":
                    top_mean,

                "Bottom_Mean_Return":
                    bottom_mean,

                "Spread":
                    spread,

                "Top_Win_Rate":
                    top_win,

                "Days":
                    len(daily_returns),

            })

            print(
                f"{horizon}D | "
                f"Top {top_n} | "
                f"Top: {top_mean:.3f}% | "
                f"Bottom: {bottom_mean:.3f}% | "
                f"Spread: {spread:.3f}% | "
                f"Win: {top_win * 100:.2f}%"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREDICTION BUCKET ANALYSIS
# ============================================================

def bucket_analysis(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "4. PREDICTION BUCKET ANALYSIS"
    )
    print("=" * 70)

    rows = []

    for horizon in HORIZONS:

        target = (
            f"Future_Return_{horizon}D"
        )

        df = data[
            [
                "Date",
                "Symbol",
                "Prediction",
                target,
            ]
        ].dropna()

        if df.empty:

            continue

        df[
            "Percentile"
        ] = (

            df
            .groupby("Date")[
                "Prediction"
            ]
            .rank(
                pct=True
            )

        )

        df[
            "Bucket"
        ] = pd.cut(

            df[
                "Percentile"
            ],

            bins=[
                0,
                0.20,
                0.40,
                0.60,
                0.80,
                1.00,
            ],

            labels=[
                "Bottom 20%",
                "20-40%",
                "40-60%",
                "60-80%",
                "Top 20%",
            ],

            include_lowest=True,

        )

        result = (

            df

            .groupby(
                "Bucket",
                observed=True
            )

            .agg(

                Observations=(
                    target,
                    "count"
                ),

                Avg_Return=(
                    target,
                    "mean"
                ),

                Median_Return=(
                    target,
                    "median"
                ),

                Win_Rate=(
                    target,
                    lambda x:
                    (x > 0).mean()
                ),

            )

            .reset_index()

        )

        result[
            "Horizon"
        ] = horizon

        result[
            "Win_Rate"
        ] *= 100

        print(
            f"\n{horizon}D:"
        )

        print(
            result.to_string(
                index=False
            )
        )

        rows.append(
            result
        )

    if rows:

        return pd.concat(
            rows,
            ignore_index=True
        )

    return pd.DataFrame()


# ============================================================
# OPEN-EXECUTION ANALYSIS
# ============================================================

def open_execution_analysis(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "5. NEXT-OPEN EXECUTION ANALYSIS"
    )
    print("=" * 70)

    rows = []

    for horizon in HORIZONS:

        target = (
            f"Open_to_Close_{horizon}D"
        )

        if target not in data.columns:

            continue

        df = data[
            [
                "Prediction",
                target,
            ]
        ].dropna()

        if df.empty:

            continue

        pearson = df[
            "Prediction"
        ].corr(
            df[target],
            method="pearson"
        )

        spearman = df[
            "Prediction"
        ].corr(
            df[target],
            method="spearman"
        )

        rows.append({

            "Horizon":
                horizon,

            "Pearson":
                pearson,

            "Spearman":
                spearman,

            "Mean_Return":
                df[target].mean(),

            "Win_Rate":
                (
                    df[target]
                    > 0
                ).mean(),

        })

        print(
            f"{horizon}D | "
            f"Pearson: {pearson:.4f} | "
            f"Spearman: {spearman:.4f} | "
            f"Mean: {df[target].mean():.3f}% | "
            f"Win: "
            f"{(
                df[target] > 0
            ).mean() * 100:.2f}%"
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STOCK-BY-STOCK PREDICTIVE POWER
# ============================================================

def stock_analysis(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "6. STOCK-BY-STOCK PREDICTIVE POWER"
    )
    print("=" * 70)

    rows = []

    target = (
        "Future_Return_20D"
    )

    for symbol, group in (
        data.groupby("Symbol")
    ):

        df = group[
            [
                "Prediction",
                target,
            ]
        ].dropna()

        if len(df) < 20:

            continue

        pearson = df[
            "Prediction"
        ].corr(
            df[target],
            method="pearson"
        )

        spearman = df[
            "Prediction"
        ].corr(
            df[target],
            method="spearman"
        )

        top_count = max(
            1,
            min(
                3,
                len(df)
            )
        )

        top_returns = (

            df
            .sort_values(
                "Prediction",
                ascending=False
            )
            .head(top_count)[
                target
            ]

        )

        rows.append({

            "Symbol":
                symbol,

            "Observations":
                len(df),

            "Pearson":
                pearson,

            "Spearman":
                spearman,

            "Top3_Mean_Return":
                top_returns.mean(),

        })

    result = (

        pd.DataFrame(rows)

        .sort_values(
            "Spearman",
            ascending=False
        )

    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# MODEL VALUE VS RANDOM
# ============================================================

def random_baseline(
    data
):

    print("\n")
    print("=" * 70)
    print(
        "7. RANKING VS RANDOM BASELINE"
    )
    print("=" * 70)

    target = (
        "Future_Return_20D"
    )

    rows = []

    rng = np.random.default_rng(
        42
    )

    for top_n in [
        1,
        3,
        5,
    ]:

        model_returns = []

        random_returns = []

        for date, group in (
            data.groupby("Date")
        ):

            group = group[
                [
                    "Symbol",
                    "Prediction",
                    target,
                ]
            ].dropna()

            if len(group) < top_n:

                continue

            model_group = (
                group
                .sort_values(
                    "Prediction",
                    ascending=False
                )
                .head(top_n)
            )

            random_group = (
                group
                .sample(
                    n=top_n,
                    random_state=int(
                        rng.integers(
                            0,
                            1_000_000
                        )
                    )
                )
            )

            model_returns.append(
                model_group[
                    target
                ].mean()
            )

            random_returns.append(
                random_group[
                    target
                ].mean()
            )

        if not model_returns:

            continue

        model_mean = np.mean(
            model_returns
        )

        random_mean = np.mean(
            random_returns
        )

        difference = (
            model_mean
            - random_mean
        )

        print(
            f"Top {top_n}: "
            f"Model {model_mean:.3f}% | "
            f"Random {random_mean:.3f}% | "
            f"Difference {difference:.3f}%"
        )

        rows.append({

            "Top_N":
                top_n,

            "Model_Return":
                model_mean,

            "Random_Return":
                random_mean,

            "Difference":
                difference,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

def final_diagnosis(
    correlation,
    rank_ic,
    top_n,
    buckets,
    random_baseline_df
):

    print("\n")
    print("=" * 70)
    print(
        "8. V32 FINAL DIAGNOSIS"
    )
    print("=" * 70)

    if correlation.empty:

        print(
            "Insufficient correlation data."
        )

        return

    # --------------------------------------------------------
    # 20D metrics
    # --------------------------------------------------------

    row20 = correlation[
        correlation[
            "Horizon"
        ] == 20
    ]

    if not row20.empty:

        spearman20 = (
            row20[
                "Spearman"
            ].iloc[0]
        )

        print(
            f"20D Spearman: "
            f"{spearman20:.4f}"
        )

    else:

        spearman20 = np.nan

    # --------------------------------------------------------
    # Rank IC
    # --------------------------------------------------------

    ic20 = rank_ic[
        rank_ic[
            "Horizon"
        ] == 20
    ]

    if not ic20.empty:

        mean_ic20 = (
            ic20[
                "Mean_Rank_IC"
            ].iloc[0]
        )

        positive_ic20 = (
            ic20[
                "Positive_IC_Rate"
            ].iloc[0]
        )

        print(
            f"20D Mean Rank IC: "
            f"{mean_ic20:.4f}"
        )

        print(
            f"20D Positive IC rate: "
            f"{positive_ic20 * 100:.2f}%"
        )

    else:

        mean_ic20 = np.nan
        positive_ic20 = np.nan

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    top3 = top_n[
        (
            top_n["Horizon"] == 20
        )
        &
        (
            top_n["Top_N"] == 3
        )
    ]

    if not top3.empty:

        top3_return = (
            top3[
                "Top_Mean_Return"
            ].iloc[0]
        )

        top3_spread = (
            top3[
                "Spread"
            ].iloc[0]
        )

        print(
            f"20D Top 3 return: "
            f"{top3_return:.3f}%"
        )

        print(
            f"20D Top3-Bottom3 spread: "
            f"{top3_spread:.3f}%"
        )

    else:

        top3_return = np.nan
        top3_spread = np.nan

    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    random3 = random_baseline_df[
        random_baseline_df[
            "Top_N"
        ] == 3
    ]

    if not random3.empty:

        difference = (
            random3[
                "Difference"
            ].iloc[0]
        )

        print(
            f"Top 3 vs random: "
            f"{difference:.3f}%"
        )

    else:

        difference = np.nan

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    print("\nConclusion:")

    if (
        pd.notna(mean_ic20)
        and mean_ic20 > 0.05
        and pd.notna(difference)
        and difference > 0
    ):

        print(
            "MODEL SHOWS PROMISING "
            "CROSS-SECTIONAL PREDICTIVE POWER."
        )

    elif (
        pd.notna(mean_ic20)
        and mean_ic20 > 0
        and pd.notna(difference)
        and difference > 0
    ):

        print(
            "MODEL SHOWS WEAK BUT POSITIVE "
            "PREDICTIVE POWER."
        )

    else:

        print(
            "MODEL DOES NOT SHOW STRONG "
            "ROBUST CROSS-SECTIONAL PREDICTIVE POWER."
        )

    print(
        "\nUse this result to decide whether "
        "V33 should modify the trading strategy "
        "or reconsider the ranking signal."
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
        "V32 - COMPLETE PREDICTION QUALITY AUDIT"
    )

    print("=" * 70)

    predictions = load_predictions()

    market_data = load_market_data()

    print(
        f"\nPrediction rows: "
        f"{len(predictions):,}"
    )

    print(
        f"Market rows: "
        f"{len(market_data):,}"
    )

    merged = merge_data(
        predictions,
        market_data
    )

    print(
        f"Merged rows: "
        f"{len(merged):,}"
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    correlation = (
        overall_correlation(
            merged
        )
    )

    rank_ic_summary, rank_ic_daily = (
        rank_ic_analysis(
            merged
        )
    )

    top_n = top_n_analysis(
        merged
    )

    buckets = bucket_analysis(
        merged
    )

    open_execution = (
        open_execution_analysis(
            merged
        )
    )

    stock = stock_analysis(
        merged
    )

    random_df = random_baseline(
        merged
    )

    final_diagnosis(
        correlation,
        rank_ic_summary,
        top_n,
        buckets,
        random_df
    )

    # ========================================================
    # SAVE
    # ========================================================

    correlation.to_csv(

        OUTPUT_DIR
        / "v32_correlations.csv",

        index=False

    )

    rank_ic_summary.to_csv(

        OUTPUT_DIR
        / "v32_rank_ic_summary.csv",

        index=False

    )

    rank_ic_daily.to_csv(

        OUTPUT_DIR
        / "v32_rank_ic_daily.csv",

        index=False

    )

    top_n.to_csv(

        OUTPUT_DIR
        / "v32_top_n_analysis.csv",

        index=False

    )

    buckets.to_csv(

        OUTPUT_DIR
        / "v32_prediction_buckets.csv",

        index=False

    )

    open_execution.to_csv(

        OUTPUT_DIR
        / "v32_open_execution.csv",

        index=False

    )

    stock.to_csv(

        OUTPUT_DIR
        / "v32_stock_analysis.csv",

        index=False

    )

    random_df.to_csv(

        OUTPUT_DIR
        / "v32_random_baseline.csv",

        index=False

    )

    merged.to_csv(

        OUTPUT_DIR
        / "v32_complete_dataset.csv",

        index=False

    )

    print("\n")
    print("=" * 70)

    print(
        "FILES SAVED"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print("=" * 70)

    print(
        "\nV32 COMPLETE"
    )


if __name__ == "__main__":

    main()

