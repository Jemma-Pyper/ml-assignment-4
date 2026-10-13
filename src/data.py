from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "networkTraffic.csv"


def load_data():
    """Load the network traffic dataset."""

    df = pd.read_csv(
        DATA_PATH,
        na_values=["?"]
    )

    return df


def prepare_data(df):
    """Prepare the predictors, target and data split."""

    X = df.drop(columns=["id", "attack_cat"])
    y = df["attack_cat"]

    nominal_cols = [
        "proto",
        "service",
        "state"
    ]

    numeric_cols = [
        col for col in X.columns
        if col not in nominal_cols
    ]

    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X,
        y,
        train_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols
    )


def main():
    df = load_data()

    (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols
    ) = prepare_data(df)

    print("Full dataset shape:", df.shape)
    print("Number of predictors:", X_tune.shape[1])
    print("Number of numeric predictors:", len(numeric_cols))
    print("Nominal predictors:", nominal_cols)

    print("\nTuning sample shape:", X_tune.shape)
    print("Evaluation sample shape:", X_eval.shape)

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nTuning class counts:")
    print(y_tune.value_counts().sort_index())

    print("\nEvaluation class counts:")
    print(y_eval.value_counts().sort_index())


if __name__ == "__main__":
    main()