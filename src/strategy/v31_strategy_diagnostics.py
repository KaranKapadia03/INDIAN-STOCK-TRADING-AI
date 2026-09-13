
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTION_FILE = Path(
    "data/models/v29_realistic_backtest/"
    "v29_predictions.csv"
)

TRADE_FILE = Path(
    "data/models/v30_daily_validation/"
    "v30_trade_log.csv"
)

OUTPUT_DIR = Path(
    "data/models/v31_strategy_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(df):

    df = df.copy()

    for column in [
        "Date",
        "Entry_Date",
        "Exit_Date",
        "Train_End",
    ]:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            try:

                if df[column].dt.tz is not None:

                    df[column] = (
                        df[column]
                        .dt.tz_localize(None)
                    )

            except Exception:

                pass

    return df


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Missing prediction file:\n"
            f"{PREDICTION_FILE}"
        )

    if not TRADE_FILE.exists():

        raise FileNotFoundError(
            f"Missing trade file:\n"
            f"{TRADE_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    trades = pd.read_csv(
        TRADE_FILE
    )

    predictions = normalize_dates(
        predictions
    )

    trades = normalize_dates(
        trades
    )

    # --------------------------------------------------------
    # Make trade return percentage available globally.
    # --------------------------------------------------------

    if "Return" in trades.columns:

        trades[
            "Return_Pct"
        ] = (
            trades[
                "Return"
            ]
            * 100
        )

    return predictions, trades


# ============================================================
# 1. PREDICTION QUALITY
# ============================================================

def prediction_quality(
    predictions
):

    print("\n")
    print("=" * 70)
    print(
        "1. PREDICTION QUALITY"
    )
    print("=" * 70)

    if (
        "Future_Return_20D"
        not in predictions.columns
    ):

        print(
            "Future_Return_20D is not present "
            "in v29_predictions.csv."
        )

        print(
            "Skipping prediction-quality analysis."
        )

        return pd.DataFrame()

    df = predictions.copy()

    df[
        "Actual_Return"
    ] = df[
        "Future_Return_20D"
    ]

    df = df.dropna(
        subset=[
            "Prediction",
            "Actual_Return",
        ]
    )

    if df.empty:

        return pd.DataFrame()

    pearson = df[
        "Prediction"
    ].corr(
        df[
            "Actual_Return"
        ],
        method="pearson"
    )

    spearman = df[
        "Prediction"
    ].corr(
        df[
            "Actual_Return"
        ],
        method="spearman"
    )

    print(
        f"Pearson correlation: "
        f"{pearson:.4f}"
    )

    print(
        f"Spearman correlation: "
        f"{spearman:.4f}"
    )

    rows = []

    for date, group in (
        df.groupby("Date")
    ):

        group = (
            group
            .sort_values(
                "Prediction",
                ascending=False
            )
        )

        if len(group) < 3:

            continue

        top1 = (
            group
            .head(1)[
                "Actual_Return"
            ]
            .mean()
        )

        top3 = (
            group
            .head(3)[
                "Actual_Return"
            ]
            .mean()
        )

        bottom3 = (
            group
            .tail(3)[
                "Actual_Return"
            ]
            .mean()
        )

        median = (
            group[
                "Actual_Return"
            ].median()
        )

        rows.append({

            "Date":
                date,

            "Top1":
                top1,

            "Top3":
                top3,

            "Bottom3":
                bottom3,

            "Median":
                median,

            "Spread":
                top3 - bottom3,

        })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    print(
        f"\nAverage Top 1 return: "
        f"{result['Top1'].mean():.3f}%"
    )

    print(
        f"Average Top 3 return: "
        f"{result['Top3'].mean():.3f}%"
    )

    print(
        f"Average Bottom 3 return: "
        f"{result['Bottom3'].mean():.3f}%"
    )

    print(
        f"Average Top3-Bottom3 spread: "
        f"{result['Spread'].mean():.3f}%"
    )

    return result


# ============================================================
# 2. PREDICTION BUCKETS
# ============================================================

def prediction_buckets(
    predictions
):

    print("\n")
    print("=" * 70)
    print(
        "2. PREDICTION BUCKET ANALYSIS"
    )
    print("=" * 70)

    if (
        "Future_Return_20D"
        not in predictions.columns
    ):

        print(
            "Future_Return_20D unavailable."
        )

        return pd.DataFrame()

    df = predictions.copy()

    df[
        "Actual_Return"
    ] = df[
        "Future_Return_20D"
    ]

    df[
        "Prediction_Percentile"
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
            "Prediction_Percentile"
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
                "Actual_Return",
                "count"
            ),

            Avg_Return=(
                "Actual_Return",
                "mean"
            ),

            Median_Return=(
                "Actual_Return",
                "median"
            ),

            Win_Rate=(
                "Actual_Return",
                lambda x:
                (x > 0).mean()
            ),

        )

        .reset_index()

    )

    result[
        "Win_Rate"
    ] *= 100

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 3. TRADE ANALYSIS
# ============================================================

def trade_analysis(
    trades
):

    print("\n")
    print("=" * 70)
    print(
        "3. TRADE ANALYSIS"
    )
    print("=" * 70)

    if trades.empty:

        print(
            "No trades."
        )

        return pd.DataFrame()

    df = trades.copy()

    if "Return" in df.columns:

        df[
            "Return_Pct"
        ] = (
            df[
                "Return"
            ]
            * 100
        )

    print(
        f"Total trades: "
        f"{len(df)}"
    )

    print(
        f"Average return: "
        f"{df['Return_Pct'].mean():.3f}%"
    )

    print(
        f"Median return: "
        f"{df['Return_Pct'].median():.3f}%"
    )

    print(
        f"Winning trades: "
        f"{(
            df['Return_Pct'] > 0
        ).sum()}"
    )

    print(
        f"Losing trades: "
        f"{(
            df['Return_Pct'] < 0
        ).sum()}"
    )

    exits = (

        df

        .groupby(
            "Exit_Reason"
        )

        .agg(

            Trades=(
                "Return_Pct",
                "count"
            ),

            Avg_Return=(
                "Return_Pct",
                "mean"
            ),

            Median_Return=(
                "Return_Pct",
                "median"
            ),

            Total_PnL=(
                "PnL",
                "sum"
            ),

            Win_Rate=(
                "Return_Pct",
                lambda x:
                (x > 0).mean()
            ),

        )

        .reset_index()

    )

    exits[
        "Win_Rate"
    ] *= 100

    print("\nExit analysis:")

    print(
        exits.to_string(
            index=False
        )
    )

    return exits


# ============================================================
# 4. STOCK FAILURE ANALYSIS
# ============================================================

def stock_failure_analysis(
    trades
):

    print("\n")
    print("=" * 70)
    print(
        "4. STOCK FAILURE ANALYSIS"
    )
    print("=" * 70)

    if trades.empty:

        return pd.DataFrame()

    df = trades.copy()

    df[
        "Return_Pct"
    ] = (
        df[
            "Return"
        ]
        * 100
    )

    result = (

        df

        .groupby(
            "Symbol"
        )

        .agg(

            Trades=(
                "Return",
                "count"
            ),

            Total_PnL=(
                "PnL",
                "sum"
            ),

            Avg_Return=(
                "Return",
                "mean"
            ),

            Median_Return=(
                "Return",
                "median"
            ),

            Win_Rate=(
                "Return",
                lambda x:
                (x > 0).mean()
            ),

            Stop_Count=(
                "Exit_Reason",
                lambda x:
                (x == "STOP").sum()
            ),

            Target_Count=(
                "Exit_Reason",
                lambda x:
                (x == "TARGET").sum()
            ),

            Time_Count=(
                "Exit_Reason",
                lambda x:
                (x == "TIME").sum()
            ),

        )

        .reset_index()

        .sort_values(
            "Total_PnL"
        )

    )

    result[
        "Avg_Return"
    ] *= 100

    result[
        "Win_Rate"
    ] *= 100

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 5. STOP ANALYSIS
# ============================================================

def stop_analysis(
    trades
):

    print("\n")
    print("=" * 70)
    print(
        "5. STOP LOSS ANALYSIS"
    )
    print("=" * 70)

    if trades.empty:

        return pd.DataFrame()

    df = trades.copy()

    df[
        "Return_Pct"
    ] = (
        df[
            "Return"
        ]
        * 100
    )

    stops = df[
        df[
            "Exit_Reason"
        ] == "STOP"
    ].copy()

    if stops.empty:

        print(
            "No stop-loss trades."
        )

        return pd.DataFrame()

    print(
        f"Stop trades: "
        f"{len(stops)}"
    )

    print(
        f"Average stop return: "
        f"{stops['Return_Pct'].mean():.3f}%"
    )

    print(
        f"Total stop PnL: "
        f"₹{stops['PnL'].sum():,.2f}"
    )

    by_stock = (

        stops

        .groupby(
            "Symbol"
        )

        .agg(

            Stops=(
                "Return",
                "count"
            ),

            Total_PnL=(
                "PnL",
                "sum"
            ),

            Avg_Return=(
                "Return",
                "mean"
            ),

        )

        .reset_index()

        .sort_values(
            "Total_PnL"
        )

    )

    by_stock[
        "Avg_Return"
    ] *= 100

    print("\nStops by stock:")

    print(
        by_stock.to_string(
            index=False
        )
    )

    return by_stock


# ============================================================
# 6. PROFIT CONCENTRATION
# ============================================================

def concentration_analysis(
    trades
):

    print("\n")
    print("=" * 70)
    print(
        "6. PROFIT CONCENTRATION"
    )
    print("=" * 70)

    if trades.empty:

        return pd.DataFrame()

    pnl = (

        trades

        .groupby(
            "Symbol"
        )[
            "PnL"
        ]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    positive = pnl[
        pnl > 0
    ]

    negative = pnl[
        pnl < 0
    ]

    total_positive = (
        positive.sum()
    )

    total_negative = (
        negative.sum()
    )

    print(
        f"Positive stock PnL: "
        f"₹{total_positive:,.2f}"
    )

    print(
        f"Negative stock PnL: "
        f"₹{total_negative:,.2f}"
    )

    if total_positive > 0:

        top1_share = (
            positive.iloc[0]
            / total_positive
        )

        top3_share = (
            positive.head(3).sum()
            / total_positive
        )

        print(
            f"Top 1 winner contribution: "
            f"{top1_share * 100:.2f}%"
        )

        print(
            f"Top 3 winner contribution: "
            f"{top3_share * 100:.2f}%"
        )

    print("\nTop contributors:")

    print(
        positive.head(5).to_string()
    )

    print("\nWorst contributors:")

    print(
        negative.tail(5).to_string()
    )

    return pnl.reset_index(
        name="PnL"
    )


# ============================================================
# 7. MARKET REGIME ANALYSIS
# ============================================================

def regime_analysis(
    predictions,
    trades
):

    print("\n")
    print("=" * 70)
    print(
        "7. MARKET REGIME ANALYSIS"
    )
    print("=" * 70)

    required = [
        "Date",
        "NIFTY_Return_20D",
        "BANKNIFTY_Return_20D",
    ]

    missing = [
        x
        for x in required
        if x not in predictions.columns
    ]

    if missing:

        print(
            f"Missing regime columns: "
            f"{missing}"
        )

        return pd.DataFrame()

    market = (

        predictions[
            required
        ]

        .drop_duplicates(
            "Date"
        )

        .copy()

    )

    market[
        "Regime"
    ] = np.select(

        [

            (
                market[
                    "NIFTY_Return_20D"
                ] > 0
            )
            &
            (
                market[
                    "BANKNIFTY_Return_20D"
                ] > 0
            ),

            (
                market[
                    "NIFTY_Return_20D"
                ] < 0
            )
            &
            (
                market[
                    "BANKNIFTY_Return_20D"
                ] < 0
            ),

        ],

        [
            "BULLISH",
            "BEARISH",
        ],

        default="MIXED",

    )

    if trades.empty:

        return market

    trade_df = trades.copy()

    trade_df[
        "Date"
    ] = trade_df[
        "Entry_Date"
    ]

    merged = trade_df.merge(
        market[
            [
                "Date",
                "Regime",
            ]
        ],
        on="Date",
        how="left"
    )

    result = (

        merged

        .groupby(
            "Regime"
        )

        .agg(

            Trades=(
                "Return",
                "count"
            ),

            Avg_Return=(
                "Return",
                "mean"
            ),

            Total_PnL=(
                "PnL",
                "sum"
            ),

            Win_Rate=(
                "Return",
                lambda x:
                (x > 0).mean()
            ),

        )

        .reset_index()

    )

    result[
        "Avg_Return"
    ] *= 100

    result[
        "Win_Rate"
    ] *= 100

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 8. CORRELATION OF PREDICTION WITH RETURNS
# ============================================================

def correlation_analysis(
    predictions
):

    print("\n")
    print("=" * 70)
    print(
        "8. CROSS-SECTIONAL CORRELATION"
    )
    print("=" * 70)

    if (
        "Future_Return_20D"
        not in predictions.columns
    ):

        print(
            "Future_Return_20D unavailable."
        )

        return pd.DataFrame()

    rows = []

    for date, group in (
        predictions.groupby(
            "Date"
        )
    ):

        group = group.dropna(
            subset=[
                "Prediction",
                "Future_Return_20D",
            ]
        )

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
            group[
                "Future_Return_20D"
            ].nunique()
            < 2
        ):

            continue

        ic = group[
            "Prediction"
        ].corr(
            group[
                "Future_Return_20D"
            ],
            method="spearman"
        )

        if pd.notna(ic):

            rows.append({

                "Date":
                    date,

                "Rank_IC":
                    ic,

            })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    print(
        f"Mean Rank IC: "
        f"{result['Rank_IC'].mean():.4f}"
    )

    print(
        f"Median Rank IC: "
        f"{result['Rank_IC'].median():.4f}"
    )

    print(
        f"Positive IC rate: "
        f"{(
            result['Rank_IC'] > 0
        ).mean() * 100:.2f}%"
    )

    return result


# ============================================================
# 9. FINAL DIAGNOSTIC SUMMARY
# ============================================================

def final_summary(
    predictions,
    trades,
    correlation,
    stocks,
    stops,
    regimes
):

    print("\n")
    print("=" * 70)
    print(
        "9. FINAL DIAGNOSTIC SUMMARY"
    )
    print("=" * 70)

    conclusions = []

    # --------------------------------------------------------
    # Prediction signal
    # --------------------------------------------------------

    if not correlation.empty:

        mean_ic = (
            correlation[
                "Rank_IC"
            ].mean()
        )

        if mean_ic > 0.10:

            conclusions.append(
                "Prediction ranking has a meaningful "
                "positive cross-sectional relationship."
            )

        elif mean_ic > 0:

            conclusions.append(
                "Prediction ranking has a weak positive "
                "cross-sectional relationship."
            )

        else:

            conclusions.append(
                "Prediction ranking has a negative "
                "cross-sectional relationship."
            )

    # --------------------------------------------------------
    # Stops
    # --------------------------------------------------------

    if not stops.empty:

        stop_pnl = (
            stops[
                "Total_PnL"
            ].sum()
        )

        conclusions.append(
            f"Stop-loss trades generated "
            f"₹{stop_pnl:,.0f} of PnL."
        )

    # --------------------------------------------------------
    # Worst stock
    # --------------------------------------------------------

    if not stocks.empty:

        worst = stocks.iloc[0]

        best = stocks.iloc[-1]

        conclusions.append(
            f"Worst stock: "
            f"{worst['Symbol']} "
            f"(₹{worst['Total_PnL']:,.0f})."
        )

        conclusions.append(
            f"Best stock: "
            f"{best['Symbol']} "
            f"(₹{best['Total_PnL']:,.0f})."
        )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    if not regimes.empty:

        best_regime = (
            regimes
            .sort_values(
                "Avg_Return",
                ascending=False
            )
            .iloc[0]
        )

        conclusions.append(
            f"Best regime by average trade: "
            f"{best_regime['Regime']}."
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if not conclusions:

        print(
            "No diagnostic conclusions available."
        )

    else:

        for i, conclusion in enumerate(
            conclusions,
            1
        ):

            print(
                f"{i}. {conclusion}"
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
        "V31 - STRATEGY FAILURE ANALYSIS"
    )

    print("=" * 70)

    predictions, trades = (
        load_data()
    )

    print(
        f"\nPredictions: "
        f"{len(predictions):,}"
    )

    print(
        f"Trades: "
        f"{len(trades):,}"
    )

    # --------------------------------------------------------
    # Run diagnostics
    # --------------------------------------------------------

    ranking = prediction_quality(
        predictions
    )

    buckets = prediction_buckets(
        predictions
    )

    exits = trade_analysis(
        trades
    )

    stocks = stock_failure_analysis(
        trades
    )

    stops = stop_analysis(
        trades
    )

    concentration = (
        concentration_analysis(
            trades
        )
    )

    regimes = regime_analysis(
        predictions,
        trades
    )

    correlation = (
        correlation_analysis(
            predictions
        )
    )

    final_summary(
        predictions,
        trades,
        correlation,
        stocks,
        stops,
        regimes
    )

    # ========================================================
    # SAVE
    # ========================================================

    outputs = {

        "v31_ranking_analysis.csv":
            ranking,

        "v31_prediction_buckets.csv":
            buckets,

        "v31_exit_analysis.csv":
            exits,

        "v31_stock_analysis.csv":
            stocks,

        "v31_stop_analysis.csv":
            stops,

        "v31_concentration.csv":
            concentration,

        "v31_regime_analysis.csv":
            regimes,

        "v31_cross_sectional_ic.csv":
            correlation,

    }

    for filename, df in (
        outputs.items()
    ):

        if (
            df is not None
            and not df.empty
        ):

            df.to_csv(
                OUTPUT_DIR / filename,
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
        "\nV31 COMPLETE"
    )


if __name__ == "__main__":

    main()
