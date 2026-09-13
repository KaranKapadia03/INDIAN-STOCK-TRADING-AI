import os
import sys

import joblib
import pandas as pd

# Allow imports from src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from src.strategy.technical_signal import generate_technical_signal
from src.strategy.risk_engine import assess_risk


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(DATA_DIR, "models")
NEWS_DIR = os.path.join(PROCESSED_DIR, "news")


# ============================================================
# SIGNAL SETTINGS
# ============================================================

ML_WEIGHT = 0.50
TECHNICAL_WEIGHT = 0.30
NEWS_WEIGHT = 0.20

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15

BUY_ML_THRESHOLD = 0.60
SELL_ML_THRESHOLD = 0.40


# ============================================================
# LOAD NEWS
# ============================================================

def load_news_signals():

    path = os.path.join(
        NEWS_DIR,
        "live_news_signals.csv"
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# CALCULATE FINAL SCORE
# ============================================================

def calculate_final_score(
    ml_probability,
    technical_score,
    news_score
):

    ml_score = (
        ml_probability * 2
    ) - 1

    technical_normalized = (
        technical_score / 6
    )

    final_score = (
        ml_score * ML_WEIGHT
        + technical_normalized * TECHNICAL_WEIGHT
        + news_score * NEWS_WEIGHT
    )

    return final_score


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(
    ml_probability,
    final_score
):

    if (
        ml_probability >= BUY_ML_THRESHOLD
        and final_score > BUY_THRESHOLD
    ):
        return "BUY"

    elif (
        ml_probability <= SELL_ML_THRESHOLD
        and final_score < SELL_THRESHOLD
    ):
        return "SELL"

    return "HOLD"


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(
    symbol,
    company,
    news_df
):

    feature_file = os.path.join(
        PROCESSED_DIR,
        symbol.replace(".", "_") + "_features_news.csv"
    )

    model_file = os.path.join(
    MODEL_DIR,
    symbol.replace(".", "_"),
    "model.pkl"
)

    if not os.path.exists(feature_file):
        return None

    if not os.path.exists(model_file):
        return None

    data = pd.read_csv(
        feature_file,
        parse_dates=["date"]
    )

    model = joblib.load(
        model_file
    )

    # --------------------------------------------------------
    # Latest market data
    # --------------------------------------------------------

    latest = data.iloc[-1]

    price = latest["Close"]

    # --------------------------------------------------------
    # Technical signal
    # --------------------------------------------------------

    technical_result = generate_technical_signal(
        data
    )

    technical_score = (
        technical_result["score"]
    )

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in model.feature_names_in_
        if column in data.columns
    ]

    X_latest = (
        data[feature_columns]
        .iloc[[-1]]
    )

    ml_probability = model.predict_proba(
        X_latest
    )[0][1]

    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    news_score = 0.0
    news_articles = 0
    news_label = "NEUTRAL"

    if not news_df.empty:

        stock_news = news_df[
        news_df["symbol"] == symbol
    ]

    if not stock_news.empty:

        row = stock_news.iloc[0]

        news_score = float(
            row.get(
                "News_Signal",
                0.0
            )
        )

        news_articles = int(
            row.get(
                "News_Count",
                0
            )
        )

        news_label = row.get(
            "News_Label",
            "NEUTRAL"
        )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_score = calculate_final_score(
        ml_probability=ml_probability,
        technical_score=technical_score,
        news_score=news_score
    )

    signal = generate_signal(
        ml_probability=ml_probability,
        final_score=final_score
    )

    # --------------------------------------------------------
    # Risk assessment
    # --------------------------------------------------------

    risk = assess_risk(
        data=data,
        ml_confidence=float(ml_probability),
        signal_strength=float(final_score),
        entry_price=float(price)
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "symbol": symbol,
        "company": company,
        "price": float(price),

        "ml_probability": float(
            ml_probability
        ),

        "technical_score": float(
            technical_score
        ),

        "news_score": float(
            news_score
        ),

        "news_articles": news_articles,

        "news_label": news_label,

        "final_score": float(
            final_score
        ),

        "signal": signal,

        "volatility": float(
            risk["volatility"]
        ),

        "atr_percent": float(
            risk["atr_percent"]
        ),

        "position_size": float(
            risk["position_size"]
        ),

        "position_size_percent": float(
            risk["position_size_percent"]
        ),

        "entry_price": float(
            risk["entry_price"]
        ),

        "stop_loss": float(
            risk["stop_loss"]
        ),

        "take_profit": float(
            risk["take_profit"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("FINAL SIGNAL + RISK ENGINE")
    print("=" * 80)

    news_df = load_news_signals()

    stocks = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "INFY.NS": "Infosys",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ITC.NS": "ITC",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "LT.NS": "Larsen & Toubro",
        "AXISBANK.NS": "Axis Bank",
        "MARUTI.NS": "Maruti Suzuki",
        "SUNPHARMA.NS": "Sun Pharmaceutical",
        "TITAN.NS": "Titan Company",
        "ADANIENT.NS": "Adani Enterprises",
        "ADANIPORTS.NS": "Adani Ports",
        "BAJFINANCE.NS": "Bajaj Finance",
        "ASIANPAINT.NS": "Asian Paints",
        "ULTRACEMCO.NS": "UltraTech Cement",
    }

    results = []

    for symbol, company in stocks.items():

        try:

            result = process_stock(
                symbol=symbol,
                company=company,
                news_df=news_df
            )

            if result is not None:
                results.append(result)

        except Exception as e:

            print(
                f"Error processing {symbol}: {e}"
            )

    if not results:

        print("No stock signals generated.")

        return

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "final_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Display rankings
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CURRENT STOCK RANKINGS")
    print("=" * 80)

    print(
        results_df[
            [
                "symbol",
                "price",
                "ml_probability",
                "technical_score",
                "news_score",
                "news_articles",
                "news_label",
                "final_score",
                "signal",
                "position_size_percent",
                "stop_loss",
                "take_profit",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BUY SIGNALS")
    print("=" * 80)

    buys = results_df[
        results_df["signal"] == "BUY"
    ]

    if buys.empty:
        print("No BUY signals.")

    else:
        print(
            buys[
                [
                    "symbol",
                    "price",
                    "ml_probability",
                    "final_score",
                    "position_size_percent",
                    "stop_loss",
                    "take_profit",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SELL SIGNALS")
    print("=" * 80)

    sells = results_df[
        results_df["signal"] == "SELL"
    ]

    if sells.empty:
        print("No SELL signals.")

    else:
        print(
            sells[
                [
                    "symbol",
                    "price",
                    "ml_probability",
                    "final_score",
                    "position_size_percent",
                    "stop_loss",
                    "take_profit",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        MODEL_DIR,
        "current_signals.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("SIGNAL ENGINE COMPLETE")
    print("=" * 80)

    print("\nSaved to:")
    print(output_file)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()