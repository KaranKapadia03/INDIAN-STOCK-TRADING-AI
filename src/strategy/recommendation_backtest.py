import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data.stock_universe import INDIAN_STOCKS


# ============================================================
# PATHS
# ============================================================

FEATURES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)

QUALITY_FILE = os.path.join(
    MODEL_DIR,
    "model_quality.csv"
)

OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    "recommendation_backtest"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BACKTEST SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

HOLDING_DAYS = 20

POSITION_SIZE = 0.20

TRANSACTION_COST = 0.001

SLIPPAGE = 0.0005

BUY_THRESHOLD = 0.20

SELL_THRESHOLD = -0.20

MIN_BUY_PROBABILITY = 0.60

MIN_SELL_PROBABILITY = 0.60

MIN_EXPECTED_RETURN = 2.0

MAX_EXPECTED_RETURN_FOR_SELL = -2.0

# Walk-forward settings
INITIAL_TRAIN_RATIO = 0.60

TEST_WINDOW = 60

RETRAIN_EVERY = 60

HORIZON = 20


# ============================================================
# LOAD MODEL QUALITY
# ============================================================

def load_model_quality():

    if not os.path.exists(
        QUALITY_FILE
    ):
        raise FileNotFoundError(
            f"Model quality file not found:\n"
            f"{QUALITY_FILE}"
        )

    quality = pd.read_csv(
        QUALITY_FILE
    )

    return quality.set_index(
        "symbol"
    )


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_stock_data(symbol):

    path = os.path.join(
        FEATURES_DIR,
        f"{symbol.replace('.', '_')}_features_news.csv"
    )

    if not os.path.exists(path):

        print(
            f"Missing feature file: {symbol}"
        )

        return None

    df = pd.read_csv(
        path
    )

    date_column = None

    for column in df.columns:

        if column.lower() == "date":

            date_column = column

            break

    if date_column is None:

        print(
            f"No date column found: {symbol}"
        )

        return None

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df = df.sort_values(
        date_column
    ).reset_index(
        drop=True
    )

    df["_date"] = df[date_column]

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(model):

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "Model does not contain feature_names_in_."
        )

    return list(
        model.feature_names_in_
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(row):

    score = 0

    # RSI
    rsi = row.get(
        "RSI_14",
        np.nan
    )

    if pd.notna(rsi):

        if rsi < 30:

            score += 2

        elif rsi < 40:

            score += 1

        elif rsi > 70:

            score -= 2

        elif rsi > 60:

            score -= 1

    # MACD
    macd = row.get(
        "MACD",
        np.nan
    )

    macd_signal = row.get(
        "MACD_Signal",
        np.nan
    )

    if (
        pd.notna(macd)
        and
        pd.notna(macd_signal)
    ):

        if macd > macd_signal:

            score += 1

        else:

            score -= 1

    # Price vs SMA20
    close = row.get(
        "Close",
        np.nan
    )

    sma20 = row.get(
        "SMA_20",
        np.nan
    )

    if (
        pd.notna(close)
        and
        pd.notna(sma20)
    ):

        if close > sma20:

            score += 1

        else:

            score -= 1

    # SMA20 vs SMA50
    sma50 = row.get(
        "SMA_50",
        np.nan
    )

    if (
        pd.notna(sma20)
        and
        pd.notna(sma50)
    ):

        if sma20 > sma50:

            score += 1

        else:

            score -= 1

    # Bollinger Bands
    bb_upper = row.get(
        "BB_Upper",
        np.nan
    )

    bb_lower = row.get(
        "BB_Lower",
        np.nan
    )

    if (
        pd.notna(close)
        and
        pd.notna(bb_upper)
        and
        pd.notna(bb_lower)
    ):

        if close <= bb_lower:

            score += 1

        elif close >= bb_upper:

            score -= 1

    return score


# ============================================================
# NORMALIZE TECHNICAL SCORE
# ============================================================

def normalize_technical_score(score):

    return float(
        np.clip(
            score / 6,
            -1,
            1
        )
    )


# ============================================================
# ML SCORE
# ============================================================

def calculate_ml_score(
    buy_probability,
    sell_probability,
    ml_weight
):

    raw_score = (

        (buy_probability - 0.50) * 2

        -

        (sell_probability - 0.50) * 2
    )

    adjusted_score = (
        raw_score * ml_weight
    )

    return float(
        np.clip(
            adjusted_score,
            -1,
            1
        )
    )


# ============================================================
# EXPECTED RETURN SCORE
# ============================================================

def calculate_return_score(
    expected_return
):

    return float(
        np.clip(
            expected_return / 10,
            -1,
            1
        )
    )


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(
    buy_probability,
    sell_probability,
    expected_return,
    ml_weight,
    tech_score
):

    ml_score = calculate_ml_score(
        buy_probability,
        sell_probability,
        ml_weight
    )

    technical = normalize_technical_score(
        tech_score
    )

    return_score = calculate_return_score(
        expected_return
    )

    # No historical news is used.
    news_score = 0.0

    final_score = (

        ml_score * 0.45

        +

        technical * 0.30

        +

        news_score * 0.15

        +

        return_score * 0.10
    )

    # BUY
    if (
        buy_probability >= MIN_BUY_PROBABILITY
        and
        expected_return >= MIN_EXPECTED_RETURN
        and
        final_score >= BUY_THRESHOLD
    ):

        return (
            "BUY",
            final_score
        )

    # SELL
    if (
        sell_probability >= MIN_SELL_PROBABILITY
        and
        expected_return <= MAX_EXPECTED_RETURN_FOR_SELL
        and
        final_score <= SELL_THRESHOLD
    ):

        return (
            "SELL",
            final_score
        )

    return (
        "HOLD",
        final_score
    )


# ============================================================
# LOAD PRODUCTION MODELS
# ============================================================

def load_models(symbol):

    folder = os.path.join(
        MODEL_DIR,
        f"{symbol.replace('.', '_')}_trade"
    )

    buy_path = os.path.join(
        folder,
        "buy_model.pkl"
    )

    sell_path = os.path.join(
        folder,
        "sell_model.pkl"
    )

    return_path = os.path.join(
        folder,
        "return_model.pkl"
    )

    if not all(
        os.path.exists(path)
        for path in [
            buy_path,
            sell_path,
            return_path
        ]
    ):

        return None

    buy_model = joblib.load(
        buy_path
    )

    sell_model = joblib.load(
        sell_path
    )

    return_model = joblib.load(
        return_path
    )

    # Important for speed
    if hasattr(
        buy_model,
        "n_jobs"
    ):

        buy_model.n_jobs = 1

    if hasattr(
        sell_model,
        "n_jobs"
    ):

        sell_model.n_jobs = 1

    if hasattr(
        return_model,
        "n_jobs"
    ):

        return_model.n_jobs = 1

    return (
        buy_model,
        sell_model,
        return_model
    )


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def create_walk_forward_predictions(
    symbol,
    df,
    quality_row
):

    models = load_models(
        symbol
    )

    if models is None:

        print(
            f"Models missing: {symbol}"
        )

        return pd.DataFrame()

    (
        buy_model,
        sell_model,
        return_model
    ) = models

    buy_features = get_model_features(
        buy_model
    )

    sell_features = get_model_features(
        sell_model
    )

    return_features = get_model_features(
        return_model
    )

    # --------------------------------------------------------
    # Start of walk-forward testing
    # --------------------------------------------------------

    n = len(df)

    initial_train_end = int(
        n * INITIAL_TRAIN_RATIO
    )

    first_test_index = (
        initial_train_end
        +
        HORIZON
    )

    predictions = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We use the already trained production models here only
    # for generating predictions on the walk-forward dates.
    #
    # The dates before the test period are never used as
    # prediction targets.
    #
    # This is still a validation of the recommendation logic,
    # not a replacement for retraining models inside every
    # walk-forward window.
    # --------------------------------------------------------

    for i in range(
        first_test_index,
        n - HORIZON
    ):

        row = df.iloc[i]

        try:

            X_buy = pd.DataFrame(
                [
                    row[buy_features].values
                ],
                columns=buy_features
            )

            X_sell = pd.DataFrame(
                [
                    row[sell_features].values
                ],
                columns=sell_features
            )

            X_return = pd.DataFrame(
                [
                    row[return_features].values
                ],
                columns=return_features
            )

        except Exception:

            continue

        # Skip missing values
        if (
            X_buy.isna().any().any()
            or
            X_sell.isna().any().any()
            or
            X_return.isna().any().any()
        ):

            continue

        buy_probability = float(
            buy_model.predict_proba(
                X_buy
            )[0][1]
        )

        sell_probability = float(
            sell_model.predict_proba(
                X_sell
            )[0][1]
        )

        expected_return = float(
            return_model.predict(
                X_return
            )[0]
        )

        # ----------------------------------------------------
        # Model quality
        # ----------------------------------------------------

        ml_weight = float(
            quality_row["ml_weight"]
        )

        # ----------------------------------------------------
        # Technical score
        # ----------------------------------------------------

        tech_score = technical_score(
            row
        )

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        signal, final_score = generate_signal(

            buy_probability,

            sell_probability,

            expected_return,

            ml_weight,

            tech_score
        )

        # ----------------------------------------------------
        # Future return
        # ----------------------------------------------------

        entry_price = float(
            row["Close"]
        )

        exit_price = float(
            df.iloc[
                i + HORIZON
            ]["Close"]
        )

        future_return = (
            exit_price
            /
            entry_price
            - 1
        ) * 100

        predictions.append({

            "symbol": symbol,

            "date": row["_date"],

            "price": entry_price,

            "buy_probability":
                buy_probability,

            "sell_probability":
                sell_probability,

            "expected_return_20d":
                expected_return,

            "technical_score":
                tech_score,

            "ml_weight":
                ml_weight,

            "final_score":
                final_score,

            "signal":
                signal,

            "future_return_20d":
                future_return
        })

    return pd.DataFrame(
        predictions
    )


# ============================================================
# BACKTEST SIGNALS
# ============================================================

def simulate_trades(
    predictions
):

    if predictions.empty:

        return {
            "trades": 0,
            "return": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0
        }, pd.DataFrame()

    predictions = predictions.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    capital = INITIAL_CAPITAL

    trades = []

    next_available_index = 0

    for i in range(
        len(predictions)
    ):

        if i < next_available_index:

            continue

        row = predictions.iloc[i]

        if row["signal"] != "BUY":

            continue

        entry_price = float(
            row["price"]
        )

        # ----------------------------------------------------
        # Position capital
        # ----------------------------------------------------

        position_capital = (
            capital
            *
            POSITION_SIZE
        )

        # ----------------------------------------------------
        # Find exit after 20 trading observations
        # ----------------------------------------------------

        exit_index = (
            i
            +
            HOLDING_DAYS
        )

        if exit_index >= len(predictions):

            break

        exit_row = predictions.iloc[
            exit_index
        ]

        exit_price = float(
            exit_row["price"]
        )

        gross_return = (
            exit_price
            /
            entry_price
            - 1
        )

        net_return = (
            gross_return
            -
            TRANSACTION_COST
            -
            SLIPPAGE
        )

        pnl = (
            position_capital
            *
            net_return
        )

        capital += pnl

        trades.append({

            "entry_date":
                row["date"],

            "exit_date":
                exit_row["date"],

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "gross_return":
                gross_return * 100,

            "net_return":
                net_return * 100,

            "pnl":
                pnl,

            "buy_probability":
                row["buy_probability"],

            "expected_return":
                row["expected_return_20d"],

            "final_score":
                row["final_score"]
        })

        next_available_index = (
            exit_index + 1
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    trade_df = pd.DataFrame(
        trades
    )

    if trade_df.empty:

        return {
            "trades": 0,
            "return": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0
        }, trade_df

    returns = (
        trade_df["net_return"]
        /
        100
    )

    win_rate = (
        returns > 0
    ).mean() * 100

    cumulative = (
        1 + returns
    ).cumprod()

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative
        /
        running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
        *
        100
    )

    if (
        len(returns) > 1
        and
        returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            /
            returns.std()
        ) * np.sqrt(
            252 / HOLDING_DAYS
        )

    else:

        sharpe = 0.0

    result = {

        "trades":
            len(trade_df),

        "final_capital":
            capital,

        "return":
            (
                capital
                /
                INITIAL_CAPITAL
                - 1
            ) * 100,

        "win_rate":
            win_rate,

        "avg_trade_return":
            returns.mean() * 100,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe
    }

    return result, trade_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print("INDIAN STOCK TRADING AI")
    print("RECOMMENDATION ENGINE VALIDATION")
    print("=" * 110)

    quality = load_model_quality()

    stock_results = []

    all_predictions = []

    all_trades = []

    # --------------------------------------------------------
    # Process stocks
    # --------------------------------------------------------

    for symbol in INDIAN_STOCKS:

        print(
            f"\nProcessing {symbol}..."
        )

        if symbol not in quality.index:

            print(
                f"No quality data: {symbol}"
            )

            continue

        df = load_stock_data(
            symbol
        )

        if df is None:

            continue

        predictions = (
            create_walk_forward_predictions(

                symbol,

                df,

                quality.loc[symbol]
            )
        )

        if predictions.empty:

            print(
                f"No predictions: {symbol}"
            )

            continue

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        all_predictions.append(
            predictions
        )

        # ----------------------------------------------------
        # Simulate
        # ----------------------------------------------------

        result, trades = simulate_trades(
            predictions
        )

        result["symbol"] = symbol

        stock_results.append(
            result
        )

        if not trades.empty:

            trades["symbol"] = symbol

            all_trades.append(
                trades
            )

        print(
            f"{symbol:<15}"
            f" Return={result['return']:+7.2f}%"
            f" Trades={result['trades']:4}"
            f" Win={result['win_rate']:6.2f}%"
            f" DD={result['max_drawdown']:7.2f}%"
            f" Sharpe={result['sharpe']:6.2f}"
        )

    # --------------------------------------------------------
    # Build outputs
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        stock_results
    )

    predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True
        )
        if all_predictions
        else pd.DataFrame()
    )

    trades_df = (
        pd.concat(
            all_trades,
            ignore_index=True
        )
        if all_trades
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_file = os.path.join(
        OUTPUT_DIR,
        "recommendation_backtest_results.csv"
    )

    predictions_file = os.path.join(
        OUTPUT_DIR,
        "recommendation_predictions.csv"
    )

    trades_file = os.path.join(
        OUTPUT_DIR,
        "recommendation_trades.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    trades_df.to_csv(
        trades_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 110)
    print("BACKTEST SUMMARY")
    print("=" * 110)

    print(
        "\nStocks tested:",
        len(results_df)
    )

    if not results_df.empty:

        print(
            "Average return:",
            f"{results_df['return'].mean():+.2f}%"
        )

        print(
            "Median return:",
            f"{results_df['return'].median():+.2f}%"
        )

        print(
            "Average win rate:",
            f"{results_df['win_rate'].mean():.2f}%"
        )

        print(
            "Average max drawdown:",
            f"{results_df['max_drawdown'].mean():.2f}%"
        )

        print(
            "Average Sharpe:",
            f"{results_df['sharpe'].mean():.2f}"
        )

        print(
            "Total trades:",
            int(
                results_df["trades"].sum()
            )
        )

    # ========================================================
    # SIGNAL DISTRIBUTION
    # ========================================================

    if not predictions_df.empty:

        print("\n")
        print("=" * 80)
        print("SIGNAL DISTRIBUTION")
        print("=" * 80)

        print(
            predictions_df[
                "signal"
            ].value_counts()
        )

    # ========================================================
    # STOCK RESULTS
    # ========================================================

    if not results_df.empty:

        print("\n")
        print("=" * 110)
        print("STOCK RESULTS")
        print("=" * 110)

        print(
            results_df[
                [
                    "symbol",
                    "return",
                    "trades",
                    "win_rate",
                    "avg_trade_return",
                    "max_drawdown",
                    "sharpe"
                ]
            ]
            .sort_values(
                "return",
                ascending=False
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # FILES
    # ========================================================

    print("\n")
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(
        results_file
    )

    print(
        predictions_file
    )

    print(
        trades_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
    