import os
import pandas as pd


# ============================================================
# PORTFOLIO RISK SETTINGS
# ============================================================

MAX_STOCK_EXPOSURE = 0.20
MAX_PORTFOLIO_EXPOSURE = 0.80

MAX_POSITIONS = 6


# ============================================================
# LOAD CURRENT SIGNALS
# ============================================================

def load_signals():

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    path = os.path.join(
        base_dir,
        "data",
        "models",
        "current_signals.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Signal file not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# SELECT BUY SIGNALS
# ============================================================

def select_buy_candidates(data):

    buys = data[
        data["signal"] == "BUY"
    ].copy()

    buys = buys.sort_values(
        "final_score",
        ascending=False
    )

    buys = buys.head(
        MAX_POSITIONS
    )

    return buys


# ============================================================
# APPLY PORTFOLIO LIMITS
# ============================================================

def calculate_portfolio_allocation(buys):

    if buys.empty:
        return buys.copy()

    buys = buys.copy()

    # --------------------------------------------------------
    # Limit individual stock exposure
    # --------------------------------------------------------

    buys["recommended_position"] = (
        buys["position_size_percent"]
        .clip(
            upper=MAX_STOCK_EXPOSURE * 100
        )
    )

    # --------------------------------------------------------
    # Limit total portfolio exposure
    # --------------------------------------------------------

    total_exposure = (
        buys["recommended_position"]
        .sum()
    )

    if total_exposure > (
        MAX_PORTFOLIO_EXPOSURE * 100
    ):

        scaling_factor = (
            MAX_PORTFOLIO_EXPOSURE * 100
            / total_exposure
        )

        buys["recommended_position"] = (
            buys["recommended_position"]
            * scaling_factor
        )

    return buys


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def calculate_summary(all_signals, portfolio):

    total_exposure = 0.0

    if not portfolio.empty:
        total_exposure = (
            portfolio["recommended_position"]
            .sum()
        )

    cash_percent = (
        100 - total_exposure
    )

    return {
        "total_stocks": len(all_signals),
        "buy_signals": int(
            (all_signals["signal"] == "BUY").sum()
        ),
        "hold_signals": int(
            (all_signals["signal"] == "HOLD").sum()
        ),
        "sell_signals": int(
            (all_signals["signal"] == "SELL").sum()
        ),
        "positions": len(portfolio),
        "portfolio_exposure": total_exposure,
        "cash_percent": cash_percent,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDIAN STOCK TRADING AI")
    print("PORTFOLIO RISK MANAGER")
    print("=" * 80)

    # --------------------------------------------------------
    # Load signals
    # --------------------------------------------------------

    signals = load_signals()

    # --------------------------------------------------------
    # Select BUY candidates
    # --------------------------------------------------------

    buys = select_buy_candidates(
        signals
    )

    # --------------------------------------------------------
    # Apply portfolio constraints
    # --------------------------------------------------------

    portfolio = calculate_portfolio_allocation(
        buys
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = calculate_summary(
        signals,
        portfolio
    )

    # --------------------------------------------------------
    # Display portfolio
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RECOMMENDED PORTFOLIO")
    print("=" * 80)

    if portfolio.empty:

        print("No BUY positions.")

    else:

        display_columns = [
            "symbol",
            "price",
            "ml_probability",
            "final_score",
            "news_label",
            "recommended_position",
            "stop_loss",
            "take_profit",
        ]

        print(
            portfolio[
                display_columns
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Portfolio summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)

    print(
        f"Total stocks analyzed: "
        f"{summary['total_stocks']}"
    )

    print(
        f"BUY signals: "
        f"{summary['buy_signals']}"
    )

    print(
        f"HOLD signals: "
        f"{summary['hold_signals']}"
    )

    print(
        f"SELL signals: "
        f"{summary['sell_signals']}"
    )

    print(
        f"Positions selected: "
        f"{summary['positions']}"
    )

    print(
        f"Portfolio exposure: "
        f"{summary['portfolio_exposure']:.2f}%"
    )

    print(
        f"Cash: "
        f"{summary['cash_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Save portfolio
    # --------------------------------------------------------

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    output_path = os.path.join(
        base_dir,
        "data",
        "models",
        "recommended_portfolio.csv"
    )

    portfolio.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("PORTFOLIO RISK MANAGEMENT COMPLETE")
    print("=" * 80)

    print("\nSaved to:")
    print(output_path)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()