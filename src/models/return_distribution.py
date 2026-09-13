from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "return_distribution"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HORIZON = 20

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

    if date_column is None:

        date_column = df.columns[0]

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["Future_Return_20D"] = (
        df["Close"].shift(-HORIZON)
        / df["Close"]
        - 1
    ) * 100

    df.dropna(
        subset=[
            "date",
            "Future_Return_20D"
        ],
        inplace=True
    )

    return df


def calculate_statistics(df, symbol):

    returns = df[
        "Future_Return_20D"
    ]

    statistics = {
        "symbol": symbol,
        "observations": len(returns),

        "mean": returns.mean(),
        "median": returns.median(),

        "std": returns.std(),

        "min": returns.min(),
        "max": returns.max(),

        "q01": returns.quantile(0.01),
        "q05": returns.quantile(0.05),
        "q10": returns.quantile(0.10),
        "q20": returns.quantile(0.20),
        "q25": returns.quantile(0.25),

        "q50": returns.quantile(0.50),

        "q75": returns.quantile(0.75),
        "q80": returns.quantile(0.80),
        "q90": returns.quantile(0.90),
        "q95": returns.quantile(0.95),
        "q99": returns.quantile(0.99),

        "positive_pct": (
            returns.gt(0).mean() * 100
        ),

        "above_2_pct": (
            returns.gt(2).mean() * 100
        ),

        "above_3_pct": (
            returns.gt(3).mean() * 100
        ),

        "above_5_pct": (
            returns.gt(5).mean() * 100
        ),

        "above_7_pct": (
            returns.gt(7).mean() * 100
        ),

        "above_10_pct": (
            returns.gt(10).mean() * 100
        ),

        "below_minus_2_pct": (
            returns.lt(-2).mean() * 100
        ),

        "below_minus_3_pct": (
            returns.lt(-3).mean() * 100
        ),

        "below_minus_5_pct": (
            returns.lt(-5).mean() * 100
        ),

        "below_minus_7_pct": (
            returns.lt(-7).mean() * 100
        ),

        "below_minus_10_pct": (
            returns.lt(-10).mean() * 100
        ),
    }

    return statistics


def main():

    print("=" * 80)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "20-DAY RETURN DISTRIBUTION"
    )
    print("=" * 80)

    all_returns = []

    stock_statistics = []

    for symbol in STOCKS:

        print(
            f"\nProcessing {symbol}..."
        )

        try:

            df = load_stock(symbol)

            statistics = calculate_statistics(
                df,
                symbol
            )

            stock_statistics.append(
                statistics
            )

            temp = df[
                [
                    "date",
                    "Close",
                    "Future_Return_20D",
                ]
            ].copy()

            temp["symbol"] = symbol

            all_returns.append(
                temp
            )

            print(
                f"Observations: "
                f"{len(df)}"
            )

            print(
                f"Mean 20D return: "
                f"{df['Future_Return_20D'].mean():.2f}%"
            )

            print(
                f"Median 20D return: "
                f"{df['Future_Return_20D'].median():.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # Stock-level statistics
    # --------------------------------------------------------

    statistics_df = pd.DataFrame(
        stock_statistics
    )

    statistics_file = (
        OUTPUT_DIR
        / "stock_return_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_file,
        index=False
    )

    # --------------------------------------------------------
    # All observations
    # --------------------------------------------------------

    returns_df = pd.concat(
        all_returns,
        ignore_index=True
    )

    returns_file = (
        OUTPUT_DIR
        / "all_20d_returns.csv"
    )

    returns_df.to_csv(
        returns_file,
        index=False
    )

    # ========================================================
    # OVERALL DISTRIBUTION
    # ========================================================

    returns = returns_df[
        "Future_Return_20D"
    ]

    print("\n")
    print("=" * 80)
    print(
        "OVERALL 20-DAY RETURN DISTRIBUTION"
    )
    print("=" * 80)

    print(
        f"\nObservations: {len(returns):,}"
    )

    print(
        f"Mean:         {returns.mean():.3f}%"
    )

    print(
        f"Median:       {returns.median():.3f}%"
    )

    print(
        f"Std Dev:      {returns.std():.3f}%"
    )

    print(
        f"Minimum:      {returns.min():.3f}%"
    )

    print(
        f"Maximum:      {returns.max():.3f}%"
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    print("\n")
    print(
        "PERCENTILES"
    )
    print("-" * 80)

    percentiles = [
        1,
        5,
        10,
        20,
        25,
        50,
        75,
        80,
        90,
        95,
        99,
    ]

    for percentile in percentiles:

        value = returns.quantile(
            percentile / 100
        )

        print(
            f"{percentile:>2}th percentile: "
            f"{value:>8.3f}%"
        )

    # ========================================================
    # RETURN THRESHOLDS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "RETURN THRESHOLDS"
    )
    print("=" * 80)

    thresholds = [
        -10,
        -7,
        -5,
        -3,
        -2,
        0,
        2,
        3,
        5,
        7,
        10,
    ]

    threshold_results = []

    for threshold in thresholds:

        if threshold >= 0:

            percentage = (
                returns.ge(threshold)
                .mean()
                * 100
            )

            condition = (
                f">= {threshold}%"
            )

        else:

            percentage = (
                returns.le(threshold)
                .mean()
                * 100
            )

            condition = (
                f"<= {threshold}%"
            )

        threshold_results.append(
            {
                "threshold_pct":
                    threshold,

                "condition":
                    condition,

                "percentage":
                    percentage,
            }
        )

        print(
            f"{condition:>10}: "
            f"{percentage:6.2f}% "
            f"of observations"
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_file = (
        OUTPUT_DIR
        / "return_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_file,
        index=False
    )

    # ========================================================
    # PROPOSED 4-CLASS TARGET
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "PROPOSED 4-CLASS TARGET DISTRIBUTION"
    )
    print("=" * 80)

    # Candidate trading classes:
    #
    # STRONG_SELL <= -5%
    # SELL         -5% to 0%
    # HOLD          0% to +5%
    # BUY          > +5%

    conditions = [
        returns <= -5,
        (returns > -5) & (returns < 0),
        (returns >= 0) & (returns <= 5),
        returns > 5,
    ]

    labels = [
        "STRONG_SELL",
        "SELL",
        "HOLD",
        "BUY",
    ]

    target = np.select(
        conditions,
        labels,
        default="HOLD"
    )

    target_distribution = (
        pd.Series(target)
        .value_counts()
        .reindex(labels)
        .fillna(0)
    )

    target_percentage = (
        target_distribution
        / len(target)
        * 100
    )

    for label in labels:

        count = int(
            target_distribution[label]
        )

        percentage = (
            target_percentage[label]
        )

        print(
            f"{label:<12} "
            f"{count:>7,} "
            f"({percentage:>6.2f}%)"
        )

    # ========================================================
    # ALTERNATIVE 3-CLASS TARGETS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "ALTERNATIVE TARGET DISTRIBUTIONS"
    )
    print("=" * 80)

    target_scenarios = {
        "2.5pct": [
            ("SELL", returns < -2.5),
            (
                "HOLD",
                (returns >= -2.5)
                & (returns <= 2.5)
            ),
            ("BUY", returns > 2.5),
        ],

        "3pct": [
            ("SELL", returns < -3),
            (
                "HOLD",
                (returns >= -3)
                & (returns <= 3)
            ),
            ("BUY", returns > 3),
        ],

        "5pct": [
            ("SELL", returns < -5),
            (
                "HOLD",
                (returns >= -5)
                & (returns <= 5)
            ),
            ("BUY", returns > 5),
        ],

        "7pct": [
            ("SELL", returns < -7),
            (
                "HOLD",
                (returns >= -7)
                & (returns <= 7)
            ),
            ("BUY", returns > 7),
        ],
    }

    scenario_results = []

    for scenario, definitions in (
        target_scenarios.items()
    ):

        print(
            f"\n--- {scenario} ---"
        )

        for label, condition in definitions:

            count = int(
                condition.sum()
            )

            percentage = (
                count
                / len(returns)
                * 100
            )

            scenario_results.append(
                {
                    "scenario":
                        scenario,

                    "class":
                        label,

                    "observations":
                        count,

                    "percentage":
                        percentage,
                }
            )

            print(
                f"{label:<6}: "
                f"{count:>7,} "
                f"({percentage:>6.2f}%)"
            )

    scenario_df = pd.DataFrame(
        scenario_results
    )

    scenario_file = (
        OUTPUT_DIR
        / "target_scenarios.csv"
    )

    scenario_df.to_csv(
        scenario_file,
        index=False
    )

    # ========================================================
    # STOCK COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "STOCK RETURN STATISTICS"
    )
    print("=" * 80)

    print(
        statistics_df[
            [
                "symbol",
                "mean",
                "median",
                "std",
                "positive_pct",
                "above_5_pct",
                "above_10_pct",
                "below_minus_5_pct",
                "below_minus_10_pct",
            ]
        ]
        .sort_values(
            "mean",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "RETURN DISTRIBUTION ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()