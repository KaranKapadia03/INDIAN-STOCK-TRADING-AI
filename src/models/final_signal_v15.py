from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTION_FILE = Path(
    "data/models/v14_regression/v14_predictions.csv"
)

FEATURE_DIR = Path(
    "data/processed"
)

NEWS_FILE = Path(
    "data/processed/news/live_news_signals.csv"
)

OUTPUT_DIR = Path(
    "data/models/v15_final_signal"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SIGNAL WEIGHTS
# ============================================================

REGRESSION_WEIGHT = 0.40
TECHNICAL_WEIGHT = 0.25
MARKET_WEIGHT = 0.20
NEWS_WEIGHT = 0.15


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(row):

    score = 0

    # --------------------------------------------------------
    # Price vs SMA20
    # --------------------------------------------------------

    close = row.get("Close", np.nan)
    sma20 = row.get("SMA_20", np.nan)

    if pd.notna(close) and pd.notna(sma20):

        if close > sma20:
            score += 1
        else:
            score -= 1

    # --------------------------------------------------------
    # SMA20 vs SMA50
    # --------------------------------------------------------

    sma50 = row.get("SMA_50", np.nan)

    if pd.notna(sma20) and pd.notna(sma50):

        if sma20 > sma50:
            score += 1
        else:
            score -= 1

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = row.get("MACD", np.nan)
    macd_signal = row.get("MACD_Signal", np.nan)

    if pd.notna(macd) and pd.notna(macd_signal):

        if macd > macd_signal:
            score += 1
        else:
            score -= 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = row.get("RSI_14", np.nan)

    if pd.notna(rsi):

        if 50 <= rsi <= 70:
            score += 1

        elif rsi < 30:
            score += 1

        elif rsi > 75:
            score -= 1

    return score


def normalize_technical(score):

    return float(
        np.clip(score / 4.0, -1, 1)
    )


# ============================================================
# MARKET SCORE
# ============================================================

def market_score(row):

    score = 0
    signals = 0

    # --------------------------------------------------------
    # NIFTY 20D return
    # --------------------------------------------------------

    nifty_return = row.get(
        "NIFTY_Return_20D",
        np.nan
    )

    if pd.notna(nifty_return):

        signals += 1

        if nifty_return > 2:
            score += 1

        elif nifty_return < -2:
            score -= 1

    # --------------------------------------------------------
    # NIFTY SMA50 vs SMA200
    # --------------------------------------------------------

    nifty_trend = row.get(
        "NIFTY_SMA50_vs_SMA200",
        np.nan
    )

    if pd.notna(nifty_trend):

        signals += 1

        if nifty_trend > 0:
            score += 1

        elif nifty_trend < 0:
            score -= 1

    # --------------------------------------------------------
    # BANKNIFTY return
    # --------------------------------------------------------

    bank_return = row.get(
        "BANKNIFTY_Return_20D",
        np.nan
    )

    if pd.notna(bank_return):

        signals += 1

        if bank_return > 2:
            score += 1

        elif bank_return < -2:
            score -= 1

    if signals == 0:
        return 0.0

    return float(
        np.clip(score / signals, -1, 1)
    )


# ============================================================
# NEWS SCORE
# ============================================================

def get_news_score(news_df, symbol):

    """
    Safely retrieve news sentiment.

    Handles possible column names such as:
        Symbol
        symbol
        Ticker
        ticker
        Stock
        stock
    """

    if news_df is None:
        return 0.0

    if news_df.empty:
        return 0.0

    # --------------------------------------------------------
    # Find stock identifier column
    # --------------------------------------------------------

    symbol_column = None

    possible_symbol_columns = [
        "Symbol",
        "symbol",
        "Ticker",
        "ticker",
        "Stock",
        "stock",
        "Ticker_Symbol",
    ]

    for column in possible_symbol_columns:

        if column in news_df.columns:
            symbol_column = column
            break

    if symbol_column is None:

        return 0.0

    # --------------------------------------------------------
    # Normalize symbols
    # --------------------------------------------------------

    target_symbol = str(
        symbol
    ).upper().strip()

    news_symbols = (
        news_df[symbol_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    matches = news_df[
        news_symbols == target_symbol
    ].copy()

    # --------------------------------------------------------
    # If exact .NS match fails, try without .NS
    # --------------------------------------------------------

    if matches.empty:

        target_without_ns = (
            target_symbol
            .replace(".NS", "")
        )

        normalized_news_symbols = (
            news_symbols
            .str.replace(
                ".NS",
                "",
                regex=False
            )
        )

        matches = news_df[
            normalized_news_symbols
            == target_without_ns
        ].copy()

    if matches.empty:
        return 0.0

    # --------------------------------------------------------
    # Find sentiment column
    # --------------------------------------------------------

    sentiment_columns = [
        "News_Sentiment",
        "Sentiment",
        "News_Score",
        "Score",
        "sentiment",
        "news_sentiment",
        "news_score",
    ]

    sentiment_column = None

    for column in sentiment_columns:

        if column in matches.columns:

            sentiment_column = column
            break

    if sentiment_column is None:
        return 0.0

    # --------------------------------------------------------
    # Latest available sentiment
    # --------------------------------------------------------

    value = matches[
        sentiment_column
    ].iloc[-1]

    if pd.isna(value):
        return 0.0

    # --------------------------------------------------------
    # Convert sentiment
    # --------------------------------------------------------

    try:

        value = float(value)

    except (
        ValueError,
        TypeError
    ):

        return 0.0

    return float(
        np.clip(value, -1, 1)
    )


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_signal(
    final_score,
    expected_return
):

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if (
        final_score >= 0.30
        and expected_return >= 2
    ):
        return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if (
        final_score <= -0.30
        and expected_return <= -2
    ):
        return "SELL"

    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    return "HOLD"


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_score(
    final_score,
    expected_return,
    technical,
    market,
    news
):

    """
    Model confidence score.

    IMPORTANT:
    This is NOT a statistical probability.
    It represents signal strength from 0-100.
    """

    # Overall signal strength

    score_strength = min(
        abs(final_score),
        1.0
    )

    # Expected return strength

    return_strength = min(
        abs(expected_return) / 10.0,
        1.0
    )

    # Technical strength

    technical_strength = abs(
        technical
    )

    # Market strength

    market_strength = abs(
        market
    )

    # News strength

    news_strength = abs(
        news
    )

    confidence = (
        score_strength * 0.45
        + return_strength * 0.25
        + technical_strength * 0.15
        + market_strength * 0.10
        + news_strength * 0.05
    )

    return round(
        float(
            np.clip(
                confidence * 100,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# FEATURE FILE FINDER
# ============================================================

def find_feature_file(symbol):

    filename = (
        symbol
        .replace(
            ".",
            "_"
        )
        + "_features.csv"
    )

    path = FEATURE_DIR / filename

    if path.exists():
        return path

    # Fallback search

    possible_files = list(
        FEATURE_DIR.glob("*_features.csv")
    )

    for file_path in possible_files:

        name = file_path.stem

        name = (
            name
            .replace(
                "_features",
                ""
            )
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INDIAN STOCK TRADING AI"
    )
    print(
        "V15 - FINAL SIGNAL ENGINE"
    )
    print("=" * 70)

    # ========================================================
    # LOAD V14 PREDICTIONS
    # ========================================================

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"\nPrediction file not found:\n"
            f"{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    print(
        f"\nPrediction rows: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # LOAD V14 RESULTS
    # ========================================================

    results_file = (
        PREDICTION_FILE.parent
        / "v14_model_results.csv"
    )

    if not results_file.exists():

        raise FileNotFoundError(
            f"\nV14 results file not found:\n"
            f"{results_file}"
        )

    results = pd.read_csv(
        results_file
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================
    #
    # IMPORTANT:
    # For this production-style signal engine,
    # choose the model with the lowest validation RMSE.
    #
    # RMSE is more appropriate here than correlation because
    # we are using the model to estimate an actual return.
    # ========================================================

    best_models = (
        results
        .sort_values(
            [
                "Symbol",
                "RMSE"
            ],
            ascending=[
                True,
                True
            ]
        )
        .groupby(
            "Symbol"
        )
        .first()
        .reset_index()
    )

    best_model_map = dict(
        zip(
            best_models["Symbol"],
            best_models["Model"]
        )
    )

    print(
        "\nBest models by stock:"
    )

    for symbol in sorted(
        best_model_map
    ):

        print(
            f"  {symbol:<15}"
            f"{best_model_map[symbol]}"
        )

    # ========================================================
    # FILTER BEST MODEL
    # ========================================================

    predictions["Best_Model"] = (
        predictions["Symbol"]
        .map(best_model_map)
    )

    predictions = predictions[
        predictions["Model"]
        == predictions["Best_Model"]
    ].copy()

    # ========================================================
    # LOAD NEWS
    # ========================================================

    if NEWS_FILE.exists():

        news = pd.read_csv(
            NEWS_FILE
        )

        print(
            f"\nNews signals loaded: "
            f"{len(news)}"
        )

        print(
            "News columns:"
        )

        print(
            list(news.columns)
        )

    else:

        news = None

        print(
            "\nNo live news file found."
        )

        print(
            "News component = 0."
        )

    # ========================================================
    # BUILD SIGNALS
    # ========================================================

    output = []

    symbols = sorted(
        predictions[
            "Symbol"
        ].dropna().unique()
    )

    print(
        f"\nGenerating signals "
        f"for {len(symbols)} stocks..."
    )

    for symbol in symbols:

        # ----------------------------------------------------
        # Get latest V14 prediction
        # ----------------------------------------------------

        stock_predictions = (
            predictions[
                predictions["Symbol"]
                == symbol
            ]
            .sort_values("Date")
        )

        if stock_predictions.empty:
            continue

        latest_prediction = (
            stock_predictions.iloc[-1]
        )

        prediction_date = (
            latest_prediction["Date"]
        )

        expected_return = float(
            latest_prediction[
                "Predicted_Return_20D"
            ]
        )

        # ----------------------------------------------------
        # Feature file
        # ----------------------------------------------------

        feature_file = (
            find_feature_file(
                symbol
            )
        )

        if feature_file is None:

            print(
                f"Skipping {symbol}: "
                "feature file not found."
            )

            continue

        features = pd.read_csv(
            feature_file
        )

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        features = (
            features
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if features.empty:
            continue

        # ----------------------------------------------------
        # Latest market data
        # ----------------------------------------------------

        latest = features.iloc[-1]

        # ----------------------------------------------------
        # TECHNICAL
        # ----------------------------------------------------

        technical_raw = (
            technical_score(
                latest
            )
        )

        technical = (
            normalize_technical(
                technical_raw
            )
        )

        # ----------------------------------------------------
        # MARKET
        # ----------------------------------------------------

        market = market_score(
            latest
        )

        # ----------------------------------------------------
        # NEWS
        # ----------------------------------------------------

        news_score = get_news_score(
            news,
            symbol
        )

        # ----------------------------------------------------
        # REGRESSION COMPONENT
        # ----------------------------------------------------
        #
        # tanh prevents extremely large predicted returns
        # from dominating the entire signal.
        # ----------------------------------------------------

        regression_component = np.tanh(
            expected_return / 5.0
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (
            regression_component
            * REGRESSION_WEIGHT

            + technical
            * TECHNICAL_WEIGHT

            + market
            * MARKET_WEIGHT

            + news_score
            * NEWS_WEIGHT
        )

        final_score = float(
            np.clip(
                final_score,
                -1,
                1
            )
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal = classify_signal(
            final_score,
            expected_return
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = (
            confidence_score(
                final_score,
                expected_return,
                technical,
                market,
                news_score
            )
        )

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        current_price = latest.get(
            "Close",
            np.nan
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi = latest.get(
            "RSI_14",
            np.nan
        )

        # ----------------------------------------------------
        # MARKET REGIME
        # ----------------------------------------------------

        regime_score = latest.get(
            "Market_Regime_Score",
            np.nan
        )

        if pd.isna(regime_score):

            regime = "UNKNOWN"

        elif regime_score > 0.20:

            regime = "BULLISH"

        elif regime_score < -0.20:

            regime = "BEARISH"

        else:

            regime = "NEUTRAL"

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output.append({

            "Date":
                latest["Date"],

            "Prediction_Date":
                prediction_date,

            "Symbol":
                symbol,

            "Signal":
                signal,

            "Confidence":
                confidence,

            "Current_Price":
                round(
                    float(current_price),
                    2
                )
                if pd.notna(
                    current_price
                )
                else np.nan,

            "Expected_Return_20D":
                round(
                    expected_return,
                    3
                ),

            "Final_Score":
                round(
                    final_score,
                    4
                ),

            "Technical_Score":
                technical_raw,

            "Technical_Component":
                round(
                    technical,
                    4
                ),

            "Market_Score":
                round(
                    market,
                    4
                ),

            "News_Score":
                round(
                    news_score,
                    4
                ),

            "Regression_Component":
                round(
                    regression_component,
                    4
                ),

            "RSI_14":
                round(
                    float(rsi),
                    2
                )
                if pd.notna(rsi)
                else np.nan,

            "Market_Regime":
                regime,

            "Best_Model":
                best_model_map.get(
                    symbol,
                    "Unknown"
                ),
        })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    final_df = pd.DataFrame(
        output
    )

    if final_df.empty:

        print(
            "\nNo signals generated."
        )

        return

    # ========================================================
    # SORT
    # ========================================================

    final_df = (
        final_df
        .sort_values(
            "Final_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SAVE CURRENT SIGNALS
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / "current_signals.csv"
    )

    final_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SAVE BUY / SELL / HOLD FILES
    # ========================================================

    buy_df = final_df[
        final_df["Signal"]
        == "BUY"
    ]

    sell_df = final_df[
        final_df["Signal"]
        == "SELL"
    ]

    hold_df = final_df[
        final_df["Signal"]
        == "HOLD"
    ]

    buy_df.to_csv(
        OUTPUT_DIR / "buy_signals.csv",
        index=False
    )

    sell_df.to_csv(
        OUTPUT_DIR / "sell_signals.csv",
        index=False
    )

    hold_df.to_csv(
        OUTPUT_DIR / "hold_signals.csv",
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CURRENT STOCK SIGNALS"
    )
    print("=" * 70)

    display_columns = [
        "Symbol",
        "Signal",
        "Confidence",
        "Current_Price",
        "Expected_Return_20D",
        "Final_Score",
        "Technical_Score",
        "Market_Score",
        "News_Score",
        "Market_Regime",
        "Best_Model",
    ]

    print(
        final_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SIGNAL DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SIGNAL DISTRIBUTION"
    )
    print("=" * 70)

    distribution = (
        final_df[
            "Signal"
        ]
        .value_counts()
    )

    print(
        distribution
    )

    # ========================================================
    # TOP BUY SIGNALS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "TOP BUY SIGNALS"
    )
    print("=" * 70)

    top_buys = (
        final_df[
            final_df["Signal"]
            == "BUY"
        ]
        .head(10)
    )

    if top_buys.empty:

        print(
            "No BUY signals."
        )

    else:

        print(
            top_buys[
                [
                    "Symbol",
                    "Confidence",
                    "Expected_Return_20D",
                    "Final_Score",
                    "Market_Regime",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # TOP SELL SIGNALS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "TOP SELL SIGNALS"
    )
    print("=" * 70)

    top_sells = (
        final_df[
            final_df["Signal"]
            == "SELL"
        ]
        .sort_values(
            "Final_Score"
        )
        .head(10)
    )

    if top_sells.empty:

        print(
            "No SELL signals."
        )

    else:

        print(
            top_sells[
                [
                    "Symbol",
                    "Confidence",
                    "Expected_Return_20D",
                    "Final_Score",
                    "Market_Regime",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V15 SUMMARY"
    )
    print("=" * 70)

    print(
        f"Stocks analyzed : "
        f"{len(final_df)}"
    )

    print(
        f"BUY signals     : "
        f"{len(buy_df)}"
    )

    print(
        f"HOLD signals    : "
        f"{len(hold_df)}"
    )

    print(
        f"SELL signals    : "
        f"{len(sell_df)}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        output_file.resolve()
    )

    print("\nFiles created:")

    print(
        "  current_signals.csv"
    )

    print(
        "  buy_signals.csv"
    )

    print(
        "  sell_signals.csv"
    )

    print(
        "  hold_signals.csv"
    )

    print("\n")
    print("=" * 70)
    print(
        "V15 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()