import os
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

PORTFOLIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "recommended_portfolio.csv"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_portfolio():

    if not os.path.exists(PORTFOLIO_FILE):
        raise FileNotFoundError(
            "recommended_portfolio.csv not found."
        )

    portfolio = pd.read_csv(
        PORTFOLIO_FILE
    )

    errors = []

    # --------------------------------------------------------
    # 1. Check allocations
    # --------------------------------------------------------

    total_allocation = (
        portfolio["recommended_position"]
        .sum()
    )

    if total_allocation > 80:
        errors.append(
            f"Portfolio exposure too high: "
            f"{total_allocation:.2f}%"
        )

    # --------------------------------------------------------
    # 2. Check individual position size
    # --------------------------------------------------------

    if (
        portfolio["recommended_position"] > 20
    ).any():

        errors.append(
            "At least one position exceeds 20%."
        )

    # --------------------------------------------------------
    # 3. Check negative allocations
    # --------------------------------------------------------

    if (
        portfolio["recommended_position"] < 0
    ).any():

        errors.append(
            "Negative position size detected."
        )

    # --------------------------------------------------------
    # 4. Check signal
    # --------------------------------------------------------

    if "signal" in portfolio.columns:

        non_buy = portfolio[
            portfolio["signal"] != "BUY"
        ]

        if not non_buy.empty:

            errors.append(
                "Non-BUY stock found in portfolio."
            )

    # --------------------------------------------------------
    # 5. Check stop loss
    # --------------------------------------------------------

    invalid_stop = portfolio[
        portfolio["stop_loss"]
        >= portfolio["entry_price"]
    ]

    if not invalid_stop.empty:

        errors.append(
            "Invalid stop-loss detected."
        )

    # --------------------------------------------------------
    # 6. Check take profit
    # --------------------------------------------------------

    invalid_target = portfolio[
        portfolio["take_profit"]
        <= portfolio["entry_price"]
    ]

    if not invalid_target.empty:

        errors.append(
            "Invalid take-profit detected."
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("PORTFOLIO VALIDATION")
    print("=" * 80)

    print()

    print(
        f"Positions: {len(portfolio)}"
    )

    print(
        f"Portfolio exposure: "
        f"{total_allocation:.2f}%"
    )

    print(
        f"Cash: "
        f"{100 - total_allocation:.2f}%"
    )

    print()

    if errors:

        print("=" * 80)
        print("VALIDATION FAILED")
        print("=" * 80)

        for error in errors:
            print("ERROR:", error)

        return False

    print("=" * 80)
    print("VALIDATION PASSED")
    print("=" * 80)

    print()
    print("All portfolio risk checks passed.")

    return True


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    validate_portfolio()