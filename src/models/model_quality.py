import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# FILE PATHS
# ============================================================

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "walk_forward_trade_models",
    "walk_forward_summary.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "model_quality.csv"
)


# ============================================================
# QUALITY THRESHOLDS
# ============================================================

# BUY model
STRONG_BUY_AUC = 0.70
USABLE_BUY_AUC = 0.60
WEAK_BUY_AUC = 0.50

# SELL model
STRONG_SELL_AUC = 0.65
USABLE_SELL_AUC = 0.58
WEAK_SELL_AUC = 0.50


# ============================================================
# CLASSIFY BUY MODEL
# ============================================================

def classify_buy_model(auc):

    if pd.isna(auc):
        return "DISABLED"

    if auc >= STRONG_BUY_AUC:
        return "STRONG ML"

    if auc >= USABLE_BUY_AUC:
        return "USABLE ML"

    if auc >= WEAK_BUY_AUC:
        return "WEAK ML"

    return "DISABLED"


# ============================================================
# CLASSIFY SELL MODEL
# ============================================================

def classify_sell_model(auc):

    if pd.isna(auc):
        return "DISABLED"

    if auc >= STRONG_SELL_AUC:
        return "STRONG ML"

    if auc >= USABLE_SELL_AUC:
        return "USABLE ML"

    if auc >= WEAK_SELL_AUC:
        return "WEAK ML"

    return "DISABLED"


# ============================================================
# ML WEIGHT
# ============================================================

def calculate_ml_weight(buy_auc, sell_auc):

    if pd.isna(buy_auc):
        buy_auc = 0.50

    if pd.isna(sell_auc):
        sell_auc = 0.50

    # Use the better validated side
    best_auc = max(
        buy_auc,
        sell_auc
    )

    if best_auc >= 0.70:
        return 1.00

    if best_auc >= 0.60:
        return 0.75

    if best_auc >= 0.55:
        return 0.50

    if best_auc >= 0.50:
        return 0.25

    return 0.00


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ML MODEL QUALITY ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Check validation file
    # --------------------------------------------------------

    if not os.path.exists(VALIDATION_FILE):

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}\n\n"
            f"Run walk_forward_trade_models.py first."
        )

    # --------------------------------------------------------
    # Load validation results
    # --------------------------------------------------------

    df = pd.read_csv(
        VALIDATION_FILE
    )

    # --------------------------------------------------------
    # Classify models
    # --------------------------------------------------------

    df["buy_quality"] = (
        df["buy_auc"]
        .apply(
            classify_buy_model
        )
    )

    df["sell_quality"] = (
        df["sell_auc"]
        .apply(
            classify_sell_model
        )
    )

    # --------------------------------------------------------
    # ML influence
    # --------------------------------------------------------

    df["ml_weight"] = df.apply(
        lambda row:
        calculate_ml_weight(
            row["buy_auc"],
            row["sell_auc"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # Overall quality
    # --------------------------------------------------------

    def overall_quality(row):

        buy_auc = row["buy_auc"]
        sell_auc = row["sell_auc"]

        if pd.isna(buy_auc):
            buy_auc = 0.50

        if pd.isna(sell_auc):
            sell_auc = 0.50

        best_auc = max(
            buy_auc,
            sell_auc
        )

        if best_auc >= 0.70:
            return "STRONG ML"

        if best_auc >= 0.60:
            return "USABLE ML"

        if best_auc >= 0.50:
            return "WEAK ML"

        return "DISABLED ML"

    df["overall_quality"] = df.apply(
        overall_quality,
        axis=1
    )

    # --------------------------------------------------------
    # BUY / SELL usability
    # --------------------------------------------------------

    df["buy_enabled"] = (
        df["buy_auc"] >= USABLE_BUY_AUC
    )

    df["sell_enabled"] = (
        df["sell_auc"] >= USABLE_SELL_AUC
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_columns = [

        "symbol",

        "buy_auc",

        "sell_auc",

        "buy_quality",

        "sell_quality",

        "overall_quality",

        "ml_weight",

        "buy_enabled",

        "sell_enabled",

        "high_buy_predictions",

        "high_buy_avg_return",

        "high_buy_win_rate",

        "high_sell_predictions",

        "high_sell_avg_return",

        "high_sell_win_rate",
    ]

    output = df[
        output_columns
    ].copy()

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 110)
    print("MODEL QUALITY BY STOCK")
    print("=" * 110)

    display_columns = [

        "symbol",

        "buy_auc",

        "sell_auc",

        "buy_quality",

        "sell_quality",

        "overall_quality",

        "ml_weight",
    ]

    print(
        output[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    print(
        "\nBUY models:"
    )

    print(
        output[
            "buy_quality"
        ].value_counts()
    )

    print(
        "\nSELL models:"
    )

    print(
        output[
            "sell_quality"
        ].value_counts()
    )

    print(
        "\nOverall ML quality:"
    )

    print(
        output[
            "overall_quality"
        ].value_counts()
    )

    # ========================================================
    # STRONGEST BUY MODELS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STRONGEST BUY MODELS")
    print("=" * 70)

    strongest_buy = (
        output[
            [
                "symbol",
                "buy_auc",
                "buy_quality",
                "high_buy_predictions",
                "high_buy_avg_return",
                "high_buy_win_rate",
            ]
        ]
        .sort_values(
            "buy_auc",
            ascending=False
        )
    )

    print(
        strongest_buy.to_string(
            index=False
        )
    )

    # ========================================================
    # STRONGEST SELL MODELS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STRONGEST SELL MODELS")
    print("=" * 70)

    strongest_sell = (
        output[
            [
                "symbol",
                "sell_auc",
                "sell_quality",
                "high_sell_predictions",
                "high_sell_avg_return",
                "high_sell_win_rate",
            ]
        ]
        .sort_values(
            "sell_auc",
            ascending=False
        )
    )

    print(
        strongest_sell.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FILE SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()