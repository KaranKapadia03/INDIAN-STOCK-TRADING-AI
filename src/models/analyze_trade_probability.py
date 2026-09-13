from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "trade_quality"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "trade_probability_analysis.csv"
)


# Probability buckets
BINS = [
    0.00,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    1.01,
]

LABELS = [
    "<50%",
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
    "70-75%",
    "75%+",
]


def analyze_stock(filepath):

    symbol = (
        filepath.parent.name
        .replace("_", ".")
    )

    data = pd.read_csv(
        filepath
    )

    if data.empty:
        return []

    data["Probability_Bucket"] = pd.cut(
        data["Trade_Probability"],
        bins=BINS,
        labels=LABELS,
        right=False
    )

    results = []

    for bucket in LABELS:

        bucket_data = data[
            data["Probability_Bucket"] == bucket
        ]

        if bucket_data.empty:
            continue

        returns = bucket_data[
            "Actual_Future_Return_5D"
        ]

        profitable = (
            returns > 0
        )

        results.append(
            {
                "symbol": symbol,
                "probability_bucket": bucket,
                "observations": len(bucket_data),

                "average_return_5D":
                    returns.mean(),

                "median_return_5D":
                    returns.median(),

                "win_rate":
                    profitable.mean(),

                "average_win":
                    returns[
                        returns > 0
                    ].mean()
                    if (returns > 0).any()
                    else np.nan,

                "average_loss":
                    returns[
                        returns <= 0
                    ].mean()
                    if (returns <= 0).any()
                    else np.nan,

                "best_return":
                    returns.max(),

                "worst_return":
                    returns.min(),
            }
        )

    return results


def main():

    print("\n")
    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("TRADE PROBABILITY ANALYSIS")
    print("=" * 80)

    prediction_files = sorted(
        MODEL_DIR.glob(
            "*/predictions.csv"
        )
    )

    if not prediction_files:

        print(
            "\nERROR: No prediction files found."
        )

        return

    all_results = []

    for filepath in prediction_files:

        symbol = (
            filepath.parent.name
            .replace("_", ".")
        )

        print(
            f"\nAnalyzing {symbol}..."
        )

        try:

            results = analyze_stock(
                filepath
            )

            all_results.extend(
                results
            )

        except Exception as e:

            print(
                f"ERROR - {symbol}: {e}"
            )

    if not all_results:

        print(
            "\nNo analysis results generated."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # AGGREGATE BY PROBABILITY BUCKET
    # -----------------------------------------------------

    aggregate = (
        results_df
        .groupby(
            "probability_bucket",
            observed=True
        )
        .apply(
            lambda group: pd.Series(
                {
                    "stocks": group["symbol"].nunique(),

                    "observations":
                        group["observations"].sum(),

                    "average_return_5D":
                        np.average(
                            group[
                                "average_return_5D"
                            ],
                            weights=group[
                                "observations"
                            ]
                        ),

                    "median_return_5D":
                        group[
                            "median_return_5D"
                        ].median(),

                    "average_win_rate":
                        np.average(
                            group[
                                "win_rate"
                            ],
                            weights=group[
                                "observations"
                            ]
                        ),
                }
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("AGGREGATE PROBABILITY RESULTS")
    print("=" * 80)

    print(
        aggregate.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # STOCK-LEVEL HIGH PROBABILITY RESULTS
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("HIGH-CONFIDENCE TRADES (60%+)")
    print("=" * 80)

    high_probability = results_df[
        results_df[
            "probability_bucket"
        ].isin(
            [
                "60-65%",
                "65-70%",
                "70-75%",
                "75%+",
            ]
        )
    ]

    if high_probability.empty:

        print(
            "No observations above 60% probability."
        )

    else:

        high_probability = (
            high_probability
            .sort_values(
                "average_return_5D",
                ascending=False
            )
        )

        print(
            high_probability[
                [
                    "symbol",
                    "probability_bucket",
                    "observations",
                    "average_return_5D",
                    "win_rate",
                    "best_return",
                    "worst_return",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # 65%+ SUMMARY
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("65%+ PROBABILITY SUMMARY")
    print("=" * 80)

    very_high = results_df[
        results_df[
            "probability_bucket"
        ].isin(
            [
                "65-70%",
                "70-75%",
                "75%+",
            ]
        )
    ]

    if very_high.empty:

        print(
            "No 65%+ probability observations."
        )

    else:

        total_observations = (
            very_high[
                "observations"
            ].sum()
        )

        weighted_return = np.average(
            very_high[
                "average_return_5D"
            ],
            weights=very_high[
                "observations"
            ]
        )

        weighted_win_rate = np.average(
            very_high[
                "win_rate"
            ],
            weights=very_high[
                "observations"
            ]
        )

        print(
            f"Observations: "
            f"{total_observations}"
        )

        print(
            f"Average 5D Return: "
            f"{weighted_return:.3f}%"
        )

        print(
            f"Win Rate: "
            f"{weighted_win_rate * 100:.2f}%"
        )

    print("\n")
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()