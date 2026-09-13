from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "strategy_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INITIAL_CAPITAL = 100000.0

INITIAL_TRAIN_RATIO = 0.60

RETRAIN_EVERY = 60

HOLDING_PERIOD = 5

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

BUY_THRESHOLD = 0.60

FINAL_THRESHOLD = 0.15

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


# ============================================================
# LOAD DATA
# ============================================================

def load_stock(symbol):

    path = (
        FEATURES_DIR
        / f"{symbol.replace('.', '_')}_features_news.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

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
# TECHNICAL SIGNAL
# ============================================================

def technical_signal(row):

    score = 0

    rsi = row["RSI_14"]

    if rsi < 30:
        score += 2
    elif rsi < 40:
        score += 1
    elif rsi > 70:
        score -= 2
    elif rsi > 60:
        score -= 1

    if row["MACD"] > row["MACD_Signal"]:
        score += 1
    else:
        score -= 1

    if row["Close"] > row["SMA_20"]:
        score += 1
    else:
        score -= 1

    if row["SMA_20"] > row["SMA_50"]:
        score += 1
    else:
        score -= 1

    if row["Close"] <= row["BB_Lower"]:
        score += 1
    elif row["Close"] >= row["BB_Upper"]:
        score -= 1

    return score


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_data(df):

    future_return = (
        df["Close"].shift(-HOLDING_PERIOD)
        / df["Close"]
        - 1
    )

    df["Future_Return"] = (
        future_return * 100
    )

    df["Target"] = (
        df["Future_Return"] > 0
    ).astype(int)

    df["Technical_Score"] = df.apply(
        technical_signal,
        axis=1
    )

    df["News_Score"] = (
        df["News_Sentiment"]
        .fillna(0)
    )

    if "News_Available" in df.columns:

        df["News_Score"] = np.where(
            df["News_Available"] == 1,
            df["News_Score"],
            0
        )

    return df


# ============================================================
# MODEL FEATURES
# ============================================================

def get_features(df):

    excluded = {
        "date",
        "symbol",
        "Target",
        "Future_Return",
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
# WALK-FORWARD ML PREDICTIONS
# ============================================================

def generate_ml_predictions(df):

    df = df.copy()

    features = get_features(df)

    df = df.dropna(
        subset=features + ["Future_Return"]
    ).copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    initial_size = int(
        len(df)
        * INITIAL_TRAIN_RATIO
    )

    predictions = []

    train_end = initial_size

    while train_end < len(df):

        train = df.iloc[
            :train_end
        ].copy()

        train = train.dropna(
            subset=["Target"]
        )

        if train["Target"].nunique() < 2:
            break

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train[features],
            train["Target"]
        )

        test_end = min(
            train_end + RETRAIN_EVERY,
            len(df)
        )

        test = df.iloc[
            train_end:test_end
        ].copy()

        test["ML_Probability"] = (
            model.predict_proba(
                test[features]
            )[:, 1]
        )

        predictions.append(test)

        train_end = test_end

    if not predictions:
        return pd.DataFrame()

    return pd.concat(
        predictions,
        ignore_index=True
    )


# ============================================================
# SIMULATE INDIVIDUAL STRATEGY
# ============================================================

def simulate_strategy(
    df,
    signal_column
):

    capital = INITIAL_CAPITAL

    trades = []

    i = 0

    while i < len(df):

        row = df.iloc[i]

        if row[signal_column] != 1:

            i += 1
            continue

        entry_price = (
            float(row["Close"])
            * (1 + SLIPPAGE)
        )

        exit_index = min(
            i + HOLDING_PERIOD,
            len(df) - 1
        )

        exit_row = df.iloc[
            exit_index
        ]

        exit_price = (
            float(exit_row["Close"])
            * (1 - SLIPPAGE)
        )

        gross_return = (
            exit_price
            / entry_price
            - 1
        )

        net_return = (
            gross_return
            - TRANSACTION_COST * 2
        )

        capital *= (
            1 + net_return
        )

        trades.append(
            {
                "entry_date":
                    row["date"],
                "exit_date":
                    exit_row["date"],
                "entry_price":
                    entry_price,
                "exit_price":
                    exit_price,
                "return":
                    net_return,
            }
        )

        i = exit_index + 1

    if not trades:

        return {
            "final_capital":
                INITIAL_CAPITAL,
            "return_pct":
                0,
            "trades":
                0,
            "win_rate":
                0,
            "max_drawdown":
                0,
        }

    trades_df = pd.DataFrame(
        trades
    )

    equity = (
        INITIAL_CAPITAL
        * (
            1
            + trades_df["return"]
        ).cumprod()
    )

    peak = equity.cummax()

    drawdown = (
        equity
        / peak
        - 1
    )

    return {
        "final_capital":
            float(capital),

        "return_pct":
            (
                capital
                / INITIAL_CAPITAL
                - 1
            )
            * 100,

        "trades":
            len(trades_df),

        "win_rate":
            (
                trades_df["return"] > 0
            ).mean()
            * 100,

        "max_drawdown":
            drawdown.min()
            * 100,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("WALK-FORWARD STRATEGY COMPARISON")
    print("=" * 80)

    results = []

    for symbol in STOCKS:

        print(
            f"\nProcessing {symbol}..."
        )

        try:

            df = load_stock(symbol)

            df = prepare_data(df)

            # ------------------------------------------------
            # WALK-FORWARD ML
            # ------------------------------------------------

            wf = generate_ml_predictions(
                df
            )

            if wf.empty:

                print(
                    "No walk-forward predictions."
                )

                continue

            # ------------------------------------------------
            # 1. BUY & HOLD
            # ------------------------------------------------

            first_price = float(
                wf["Close"].iloc[0]
            )

            last_price = float(
                wf["Close"].iloc[-1]
            )

            buy_hold_return = (
                last_price
                / first_price
                - 1
            ) * 100

            # ------------------------------------------------
            # 2. TECHNICAL ONLY
            # ------------------------------------------------

            wf["Technical_Buy"] = (
                wf["Technical_Score"] >= 3
            ).astype(int)

            technical_result = (
                simulate_strategy(
                    wf,
                    "Technical_Buy"
                )
            )

            # ------------------------------------------------
            # 3. ML ONLY
            # ------------------------------------------------

            wf["ML_Buy"] = (
                wf["ML_Probability"]
                >= BUY_THRESHOLD
            ).astype(int)

            ml_result = (
                simulate_strategy(
                    wf,
                    "ML_Buy"
                )
            )

            # ------------------------------------------------
            # 4. ML + TECHNICAL
            # ------------------------------------------------

            wf["ML_Tech_Buy"] = (
                (
                    wf["ML_Probability"]
                    >= BUY_THRESHOLD
                )
                &
                (
                    wf["Technical_Score"]
                    > 0
                )
            ).astype(int)

            ml_tech_result = (
                simulate_strategy(
                    wf,
                    "ML_Tech_Buy"
                )
            )

            # ------------------------------------------------
            # 5. ML + TECHNICAL + NEWS
            # ------------------------------------------------

            wf["ML_Score"] = (
                wf["ML_Probability"]
                * 2
                - 1
            )

            wf["Technical_Normalized"] = (
                wf["Technical_Score"]
                / 6
            )

            wf["Final_Score"] = (
                wf["ML_Score"] * 0.50
                +
                wf["Technical_Normalized"]
                * 0.30
                +
                wf["News_Score"]
                * 0.20
            )

            wf["Full_Buy"] = (
                (
                    wf["ML_Probability"]
                    >= BUY_THRESHOLD
                )
                &
                (
                    wf["Final_Score"]
                    > FINAL_THRESHOLD
                )
            ).astype(int)

            full_result = (
                simulate_strategy(
                    wf,
                    "Full_Buy"
                )
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            results.append(
                {
                    "symbol":
                        symbol,

                    "buy_hold_return_pct":
                        buy_hold_return,

                    "technical_return_pct":
                        technical_result[
                            "return_pct"
                        ],

                    "ml_return_pct":
                        ml_result[
                            "return_pct"
                        ],

                    "ml_technical_return_pct":
                        ml_tech_result[
                            "return_pct"
                        ],

                    "ml_technical_news_return_pct":
                        full_result[
                            "return_pct"
                        ],

                    "technical_trades":
                        technical_result[
                            "trades"
                        ],

                    "ml_trades":
                        ml_result[
                            "trades"
                        ],

                    "ml_technical_trades":
                        ml_tech_result[
                            "trades"
                        ],

                    "full_strategy_trades":
                        full_result[
                            "trades"
                        ],

                    "full_strategy_win_rate":
                        full_result[
                            "win_rate"
                        ],

                    "full_strategy_max_drawdown":
                        full_result[
                            "max_drawdown"
                        ],
                }
            )

            print(
                f"BUY & HOLD: "
                f"{buy_hold_return:.2f}%"
            )

            print(
                f"TECHNICAL: "
                f"{technical_result['return_pct']:.2f}%"
            )

            print(
                f"ML ONLY: "
                f"{ml_result['return_pct']:.2f}%"
            )

            print(
                f"ML + TECHNICAL: "
                f"{ml_tech_result['return_pct']:.2f}%"
            )

            print(
                f"FULL: "
                f"{full_result['return_pct']:.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR: {symbol} -> {e}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR
        / "strategy_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("AVERAGE RESULTS")
    print("=" * 80)

    if not results_df.empty:

        print(
            f"\nBuy & Hold: "
            f"{results_df['buy_hold_return_pct'].mean():.2f}%"
        )

        print(
            f"Technical: "
            f"{results_df['technical_return_pct'].mean():.2f}%"
        )

        print(
            f"ML Only: "
            f"{results_df['ml_return_pct'].mean():.2f}%"
        )

        print(
            f"ML + Technical: "
            f"{results_df['ml_technical_return_pct'].mean():.2f}%"
        )

        print(
            f"ML + Technical + News: "
            f"{results_df['ml_technical_news_return_pct'].mean():.2f}%"
        )

        print("\n")
        print(
            results_df[
                [
                    "symbol",
                    "buy_hold_return_pct",
                    "technical_return_pct",
                    "ml_return_pct",
                    "ml_technical_return_pct",
                    "ml_technical_news_return_pct",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":

    main()