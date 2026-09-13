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
    "data/models/v14_regression/v14_predictions.csv"
)

RESULTS_FILE = Path(
    "data/models/v14_regression/v14_model_results.csv"
)

OUTPUT_DIR = Path(
    "data/models/v17_signal_filter"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SIGNAL CONFIG
# ============================================================

MIN_EXPECTED_RETURN = 3.0

MIN_TECHNICAL_SCORE = 2

MIN_QUALITY_SCORE = 0.15

BUY_THRESHOLD = 0.15

SELL_THRESHOLD = -0.15


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(
    df,
    column="Date"
):

    df = df.copy()

    if column not in df.columns:
        return df

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    if (
        hasattr(
            df[column].dt,
            "tz"
        )
        and df[column].dt.tz is not None
    ):

        df[column] = (
            df[column]
            .dt.tz_localize(None)
        )

    df[column] = (
        df[column]
        .dt.normalize()
    )

    return df


# ============================================================
# FEATURE FILE
# ============================================================

def find_feature_file(
    symbol
):

    filename = (
        symbol
        .replace(
            ".",
            "_"
        )
        + "_features.csv"
    )

    path = (
        FEATURE_DIR
        / filename
    )

    if path.exists():
        return path

    # Fallback search

    for file_path in FEATURE_DIR.glob(
        "*_features.csv"
    ):

        name = file_path.stem

        name = name.replace(
            "_features",
            ""
        )

        if name.endswith("_NS"):

            name = (
                name[:-3]
                + ".NS"
            )

        if name.upper() == symbol.upper():

            return file_path

    return None


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(
    row
):

    score = 0

    close = row.get(
        "Close",
        np.nan
    )

    sma20 = row.get(
        "SMA_20",
        np.nan
    )

    sma50 = row.get(
        "SMA_50",
        np.nan
    )

    macd = row.get(
        "MACD",
        np.nan
    )

    macd_signal = row.get(
        "MACD_Signal",
        np.nan
    )

    rsi = row.get(
        "RSI_14",
        np.nan
    )

    # --------------------------------------------------------
    # Price above SMA20
    # --------------------------------------------------------

    if (
        pd.notna(close)
        and pd.notna(sma20)
        and close > sma20
    ):

        score += 1

    # --------------------------------------------------------
    # SMA20 above SMA50
    # --------------------------------------------------------

    if (
        pd.notna(sma20)
        and pd.notna(sma50)
        and sma20 > sma50
    ):

        score += 1

    # --------------------------------------------------------
    # MACD bullish
    # --------------------------------------------------------

    if (
        pd.notna(macd)
        and pd.notna(macd_signal)
        and macd > macd_signal
    ):

        score += 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if pd.notna(rsi):

        if 50 <= rsi <= 70:
            score += 1

    return score


# ============================================================
# MARKET SCORE
# ============================================================

def market_score(
    row
):

    score = 0
    count = 0

    # --------------------------------------------------------
    # NIFTY 20D return
    # --------------------------------------------------------

    nifty_return = row.get(
        "NIFTY_Return_20D",
        np.nan
    )

    if pd.notna(
        nifty_return
    ):

        count += 1

        if nifty_return > 2:

            score += 1

        elif nifty_return < -2:

            score -= 1

    # --------------------------------------------------------
    # NIFTY SMA50 / SMA200
    # --------------------------------------------------------

    nifty_trend = row.get(
        "NIFTY_SMA50_vs_SMA200",
        np.nan
    )

    if pd.notna(
        nifty_trend
    ):

        count += 1

        if nifty_trend > 0:

            score += 1

        elif nifty_trend < 0:

            score -= 1

    # --------------------------------------------------------
    # BANKNIFTY
    # --------------------------------------------------------

    bank_return = row.get(
        "BANKNIFTY_Return_20D",
        np.nan
    )

    if pd.notna(
        bank_return
    ):

        count += 1

        if bank_return > 2:

            score += 1

        elif bank_return < -2:

            score -= 1

    if count == 0:

        return 0.0

    return float(
        np.clip(
            score / count,
            -1,
            1
        )
    )


# ============================================================
# MODEL QUALITY
# ============================================================

def build_model_quality(
    results
):

    quality = {}

    for symbol in results[
        "Symbol"
    ].unique():

        stock = results[
            results["Symbol"]
            == symbol
        ].copy()

        if stock.empty:
            continue

        # Select lowest RMSE model

        best = (
            stock
            .sort_values(
                "RMSE"
            )
            .iloc[0]
        )

        rmse = float(
            best["RMSE"]
        )

        correlation = float(
            best["Correlation"]
        )

        direction = float(
            best[
                "Direction_Accuracy"
            ]
        )

        # ----------------------------------------------------
        # Quality components
        # ----------------------------------------------------

        # Correlation:
        # useful prediction should have positive correlation

        corr_score = np.clip(
            correlation,
            0,
            1
        )

        # Direction:
        # 50% = random
        # 60% = meaningful
        direction_score = np.clip(
            (
                direction
                - 0.50
            )
            / 0.15,
            0,
            1
        )

        # RMSE quality
        #
        # Lower RMSE is better.
        #
        # A 5% RMSE is treated as substantially
        # better than a 10% RMSE.

        rmse_score = np.clip(
            (
                10
                - rmse
            )
            / 5,
            0,
            1
        )

        model_quality = (
            corr_score * 0.50
            + direction_score * 0.30
            + rmse_score * 0.20
        )

        quality[
            symbol
        ] = {

            "Best_Model":
                best["Model"],

            "RMSE":
                rmse,

            "Correlation":
                correlation,

            "Direction_Accuracy":
                direction,

            "Model_Quality":
                model_quality,
        }

    return quality


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_quality(
    expected_return,
    technical,
    market,
    model_quality
):

    # --------------------------------------------------------
    # Expected return component
    # --------------------------------------------------------

    return_component = np.tanh(
        expected_return / 8
    )

    # --------------------------------------------------------
    # Technical component
    # --------------------------------------------------------

    technical_component = (
        technical / 4
    )

    # --------------------------------------------------------
    # Final quality
    # --------------------------------------------------------

    quality = (

        return_component * 0.40

        + technical_component * 0.25

        + market * 0.15

        + model_quality * 0.20
    )

    return float(
        np.clip(
            quality,
            -1,
            1
        )
    )


# ============================================================
# CLASSIFY
# ============================================================

def classify_signal(
    expected_return,
    technical,
    quality
):

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if (
        expected_return
        >= MIN_EXPECTED_RETURN
        and technical
        >= MIN_TECHNICAL_SCORE
        and quality
        >= BUY_THRESHOLD
    ):

        return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if (
        expected_return
        <= -MIN_EXPECTED_RETURN
        and technical
        <= 1
        and quality
        <= SELL_THRESHOLD
    ):

        return "SELL"

    return "HOLD"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "V17 - SIGNAL QUALITY FILTER"
    )
    print("=" * 70)

    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions = normalize_dates(
        predictions
    )

    print(
        f"\nPrediction rows: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"Results file not found:\n"
            f"{RESULTS_FILE}"
        )

    results = pd.read_csv(
        RESULTS_FILE
    )

    model_quality = (
        build_model_quality(
            results
        )
    )

    print(
        f"Model quality scores: "
        f"{len(model_quality)} stocks"
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_map = {}

    for symbol in model_quality:

        best_model_map[
            symbol
        ] = model_quality[
            symbol
        ]["Best_Model"]

    predictions[
        "Best_Model"
    ] = predictions[
        "Symbol"
    ].map(
        best_model_map
    )

    predictions = predictions[
        predictions[
            "Model"
        ]
        == predictions[
            "Best_Model"
        ]
    ].copy()

    print(
        f"Rows after best-model filter: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    output = []

    symbols = sorted(
        predictions[
            "Symbol"
        ]
        .dropna()
        .unique()
    )

    print(
        f"\nProcessing "
        f"{len(symbols)} stocks..."
    )

    for symbol in symbols:

        feature_file = (
            find_feature_file(
                symbol
            )
        )

        if feature_file is None:

            print(
                f"Skipping {symbol}: "
                f"feature file not found."
            )

            continue

        features = pd.read_csv(
            feature_file
        )

        features = normalize_dates(
            features
        )

        features = (
            features
            .sort_values("Date")
            .drop_duplicates(
                subset=["Date"]
            )
            .reset_index(
                drop=True
            )
        )

        stock_predictions = (
            predictions[
                predictions[
                    "Symbol"
                ]
                == symbol
            ]
            .sort_values(
                "Date"
            )
        )

        if stock_predictions.empty:
            continue

        quality_info = (
            model_quality
            .get(
                symbol,
                {}
            )
        )

        stock_model_quality = float(
            quality_info.get(
                "Model_Quality",
                0
            )
        )

        correlation = float(
            quality_info.get(
                "Correlation",
                0
            )
        )

        rmse = float(
            quality_info.get(
                "RMSE",
                np.nan
            )
        )

        direction = float(
            quality_info.get(
                "Direction_Accuracy",
                np.nan
            )
        )

        best_model = quality_info.get(
            "Best_Model",
            "Unknown"
        )

        # ----------------------------------------------------
        # Align prediction dates with features
        # ----------------------------------------------------

        merged = pd.merge(
            stock_predictions[
                [
                    "Date",
                    "Predicted_Return_20D"
                ]
            ],
            features,
            on="Date",
            how="inner"
        )

        if merged.empty:
            continue

        # ----------------------------------------------------
        # Generate signals
        # ----------------------------------------------------

        for _, row in merged.iterrows():

            expected_return = float(
                row[
                    "Predicted_Return_20D"
                ]
            )

            technical = (
                technical_score(
                    row
                )
            )

            market = (
                market_score(
                    row
                )
            )

            quality = (
                calculate_quality(
                    expected_return,
                    technical,
                    market,
                    stock_model_quality
                )
            )

            signal = (
                classify_signal(
                    expected_return,
                    technical,
                    quality
                )
            )

            output.append({

                "Date":
                    row["Date"],

                "Symbol":
                    symbol,

                "Signal":
                    signal,

                "Expected_Return_20D":
                    expected_return,

                "Technical_Score":
                    technical,

                "Market_Score":
                    market,

                "Model_Quality":
                    stock_model_quality,

                "Model_Correlation":
                    correlation,

                "Model_RMSE":
                    rmse,

                "Model_Direction_Accuracy":
                    direction,

                "Quality_Score":
                    quality,

                "Best_Model":
                    best_model,
            })

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if not output:

        print(
            "\nERROR: No signals were generated."
        )

        print(
            "Check that prediction dates and "
            "feature dates overlap."
        )

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        output
    )

    df = normalize_dates(
        df
    )

    df = (
        df
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE ALL SIGNALS
    # ========================================================

    all_file = (
        OUTPUT_DIR
        / "v17_signals.csv"
    )

    df.to_csv(
        all_file,
        index=False
    )

    # ========================================================
    # CURRENT SIGNALS
    # ========================================================

    current = (
        df
        .sort_values(
            "Date"
        )
        .groupby(
            "Symbol"
        )
        .tail(1)
        .sort_values(
            "Quality_Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    current_file = (
        OUTPUT_DIR
        / "current_signals.csv"
    )

    current.to_csv(
        current_file,
        index=False
    )

    # ========================================================
    # BUY SIGNALS
    # ========================================================

    buys = df[
        df["Signal"]
        == "BUY"
    ]

    buys.to_csv(
        OUTPUT_DIR
        / "buy_signals.csv",
        index=False
    )

    # ========================================================
    # SELL SIGNALS
    # ========================================================

    sells = df[
        df["Signal"]
        == "SELL"
    ]

    sells.to_csv(
        OUTPUT_DIR
        / "sell_signals.csv",
        index=False
    )

    # ========================================================
    # HOLD SIGNALS
    # ========================================================

    holds = df[
        df["Signal"]
        == "HOLD"
    ]

    holds.to_csv(
        OUTPUT_DIR
        / "hold_signals.csv",
        index=False
    )

    # ========================================================
    # DISPLAY CURRENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CURRENT SIGNALS"
    )
    print("=" * 70)

    print(
        current[
            [
                "Symbol",
                "Signal",
                "Expected_Return_20D",
                "Technical_Score",
                "Market_Score",
                "Model_Quality",
                "Model_Correlation",
                "Quality_Score",
                "Best_Model",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # CURRENT DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CURRENT SIGNAL DISTRIBUTION"
    )
    print("=" * 70)

    print(
        current[
            "Signal"
        ].value_counts()
    )

    # ========================================================
    # HISTORICAL DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "HISTORICAL SIGNAL DISTRIBUTION"
    )
    print("=" * 70)

    print(
        df[
            "Signal"
        ].value_counts()
    )

    # ========================================================
    # STRONGEST SIGNALS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "STRONGEST BUY SIGNALS"
    )
    print("=" * 70)

    top_buy = (
        current[
            current[
                "Signal"
            ]
            == "BUY"
        ]
        .sort_values(
            "Quality_Score",
            ascending=False
        )
        .head(10)
    )

    if top_buy.empty:

        print(
            "No BUY signals."
        )

    else:

        print(
            top_buy[
                [
                    "Symbol",
                    "Expected_Return_20D",
                    "Technical_Score",
                    "Model_Quality",
                    "Quality_Score",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # STRONGEST SELL SIGNALS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "STRONGEST SELL SIGNALS"
    )
    print("=" * 70)

    top_sell = (
        current[
            current[
                "Signal"
            ]
            == "SELL"
        ]
        .sort_values(
            "Quality_Score"
        )
        .head(10)
    )

    if top_sell.empty:

        print(
            "No SELL signals."
        )

    else:

        print(
            top_sell[
                [
                    "Symbol",
                    "Expected_Return_20D",
                    "Technical_Score",
                    "Model_Quality",
                    "Quality_Score",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V17 COMPLETE"
    )
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        all_file.resolve()
    )

    print(
        current_file.resolve()
    )


if __name__ == "__main__":
    main()