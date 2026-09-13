from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "production"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_ml"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "universal_rf.joblib"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_list.joblib"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "live_predictions.csv"
)

JSON_FILE = (
    OUTPUT_DIR
    / "live_predictions.json"
)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

def load_feature_list():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature list not found:\n"
            f"{FEATURE_FILE}"
        )

    features = joblib.load(
        FEATURE_FILE
    )

    if not isinstance(
        features,
        list
    ):

        raise RuntimeError(
            "Production feature list is invalid."
        )

    return features


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Production model not found:\n"
            f"{MODEL_FILE}\n\n"
            "Run:\n"
            "python src\\models\\train_production_model.py"
        )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        model,
        "predict"
    ):

        raise RuntimeError(
            "Production model does not have "
            "a predict() method."
        )

    return model


# ============================================================
# LOAD LATEST STOCK FEATURES
# ============================================================

def load_latest_features():

    files = sorted(
        FEATURE_DIR.glob(
            "*_features.csv"
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No feature files found:\n"
            f"{FEATURE_DIR}"
        )

    rows = []

    for file in files:

        try:

            df = pd.read_csv(
                file
            )

        except Exception as exc:

            print(
                f"WARNING: Could not read "
                f"{file.name}: {exc}"
            )

            continue

        if df.empty:
            continue

        # Remove duplicate column names.

        df = df.loc[
            :,
            ~df.columns.duplicated()
        ]

        if "Date" not in df.columns:
            continue

        if "Close" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date"]
        )

        if df.empty:
            continue

        df = df.sort_values(
            "Date"
        )

        latest = (
            df.iloc[-1]
            .copy()
        )

        # Convert filename:
        #
        # RELIANCE_NS_features.csv
        #
        # to:
        #
        # RELIANCE.NS

        symbol = (
            file.stem
            .replace(
                "_features",
                ""
            )
            .replace(
                "_NS",
                ".NS"
            )
        )

        latest["Symbol"] = symbol

        rows.append(
            latest
        )

    if not rows:

        raise RuntimeError(
            "No latest stock feature rows found."
        )

    result = pd.DataFrame(
        rows
    )

    result = result.loc[
        :,
        ~result.columns.duplicated()
    ]

    result = result.sort_values(
        "Symbol"
    ).reset_index(
        drop=True
    )

    return result


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_input(
    df,
    features
):

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "\nMissing production features:\n"
            f"{missing}"
        )

    X = df[
        features
    ].copy()

    X = X.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # Cross-sectional median fallback.

    for column in features:

        median = X[
            column
        ].median()

        if pd.isna(median):

            median = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            median
        )

    return X


# ============================================================
# NORMALIZE PREDICTIONS
# ============================================================

def normalize_predictions(
    predictions
):

    values = np.asarray(
        predictions,
        dtype=float
    )

    if len(values) <= 1:

        return np.zeros(
            len(values)
        )

    mean = np.mean(
        values
    )

    std = np.std(
        values
    )

    if std == 0 or np.isnan(std):

        return np.zeros(
            len(values)
        )

    z_scores = (
        values - mean
    ) / std

    # Convert approximately to [-1, +1].

    normalized = (
        z_scores / 2.0
    )

    return np.clip(
        normalized,
        -1.0,
        1.0
    )


# ============================================================
# GENERATE LIVE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X
):

    predictions = model.predict(
        X
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INDIAN STOCK TRADING AI"
    )

    print(
        "LIVE ML PREDICTION PIPELINE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading production model..."
    )

    model = load_model()

    print(
        f"Model type: "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = (
        load_feature_list()
    )

    print(
        f"Features used: "
        f"{len(features)}"
    )

    print(
        features
    )

    # --------------------------------------------------------
    # STOCK DATA
    # --------------------------------------------------------

    print(
        "\nLoading latest market features..."
    )

    df = (
        load_latest_features()
    )

    print(
        f"Stocks loaded: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # CHECK STOCK COUNT
    # --------------------------------------------------------

    if len(df) != 20:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected 20 stocks, "
            f"found {len(df)}."
        )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    X = prepare_input(
        df,
        features
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print(
        "\nGenerating fresh ML predictions..."
    )

    predictions = (
        generate_predictions(
            model,
            X
        )
    )

    # --------------------------------------------------------
    # ADD PREDICTIONS
    # --------------------------------------------------------

    df[
        "Prediction"
    ] = predictions

    df[
        "ML_Normalized"
    ] = normalize_predictions(
        predictions
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    df[
        "ML_Rank"
    ] = (
        df[
            "Prediction"
        ]
        .rank(
            ascending=False,
            method="first"
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # PREDICTION TYPE
    # --------------------------------------------------------

    df[
        "Prediction_Type"
    ] = (
        "LIVE_20D_EXCESS_RETURN"
    )

    # --------------------------------------------------------
    # GENERATED DATE
    # --------------------------------------------------------

    df[
        "Generated_At"
    ] = pd.Timestamp.now().isoformat(
        timespec="seconds"
    )

    # --------------------------------------------------------
    # OUTPUT COLUMNS
    # --------------------------------------------------------

    output_columns = [

        "Symbol",

        "Date",

        "Close",

        "Prediction",

        "ML_Normalized",

        "ML_Rank",

        "Prediction_Type",

        "Generated_At",

    ]

    output = df[
        [
            column
            for column in output_columns
            if column in df.columns
        ]
    ].copy()

    output = output.sort_values(
        "ML_Rank"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output.to_json(
        JSON_FILE,
        orient="records",
        indent=2
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "LIVE ML STOCK RANKING"
    )

    print("=" * 70)

    display_columns = [

        "ML_Rank",

        "Symbol",

        "Close",

        "Prediction",

        "ML_Normalized",

    ]

    print(
        output[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)

    print(
        "FILES SAVED"
    )

    print("=" * 70)

    print(
        f"\nCSV:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nJSON:\n"
        f"{JSON_FILE.resolve()}"
    )

    print("\n")
    print(
        "LIVE ML PREDICTION COMPLETE"
    )


if __name__ == "__main__":

    main()