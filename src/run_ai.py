from pathlib import Path
import subprocess
import sys
import time


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PIPELINE STEPS
# ============================================================

PIPELINE = [
    (
        "LIVE MARKET DATA",
        "src/data/market_data.py",
    ),
    (
        "BUILD TECHNICAL FEATURES",
        "src/features/build_features.py",
    ),
    (
        "ADD MARKET FEATURES",
        "src/features/add_market_features.py",
    ),
    (
        "LIVE NEWS",
        "src/features/live_news_signal.py",
    ),
    (
        "LIVE ML PREDICTIONS",
        "src/models/live_predictions.py",
    ),
    (
        "AI RECOMMENDATIONS",
        "src/strategy/production_engine.py",
    ),
]


# ============================================================
# RUN SCRIPT
# ============================================================

def run_step(
    name,
    script
):

    print("\n")
    print("=" * 75)
    print(name)
    print("=" * 75)

    script_path = (
        PROJECT_ROOT
        / script
    )

    if not script_path.exists():

        print(
            f"ERROR: Script not found:\n"
            f"{script_path}"
        )

        return False

    start = time.time()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.time()
        - start
    )

    print(
        f"\nCompleted in "
        f"{elapsed:.1f} seconds."
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {name} failed."
        )

        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "INDIAN STOCK TRADING AI"
    )

    print(
        "AUTOMATED AI PIPELINE"
    )

    print("=" * 75)

    print(
        "\nPipeline:"
    )

    for index, (
        name,
        script
    ) in enumerate(
        PIPELINE,
        start=1
    ):

        print(
            f"{index}. {name}"
        )

    overall_start = time.time()

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    for name, script in PIPELINE:

        success = run_step(
            name,
            script
        )

        if not success:

            print("\n")
            print("=" * 75)

            print(
                "PIPELINE STOPPED"
            )

            print("=" * 75)

            print(
                f"\nFailed step: "
                f"{name}"
            )

            print(
                f"Script: "
                f"{script}"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - overall_start
    )

    print("\n")
    print("=" * 75)

    print(
        "AI PIPELINE COMPLETE"
    )

    print("=" * 75)

    print(
        f"\nTotal runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\nLatest recommendations:"
    )

    print(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "production"
        / "recommendations.csv"
    )

    print(
        "\nThe system has completed:"
    )

    print(
        "  [OK] Market data"
    )

    print(
        " [OK] Technical indicators"
    )

    print(
        "  [OK] Market context"
    )

    print(
        "  [OK] Live news"
    )

    print(
        "  [OK] ML predictions"
    )

    print(
        "  [OK] BUY / HOLD / SELL"
    )

    print("\n")


if __name__ == "__main__":

    main()