from pathlib import Path
from time import perf_counter

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


# ---------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 3
N_JOBS = 2

TREE_COUNTS = [
    100,
    200,
    400,
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_pipeline(
    numeric_cols,
    nominal_cols,
    n_estimators,
):
    """
    Construct the current Random Forest candidate while varying only
    the number of trees.
    """

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing",
            ),
        ),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric_cols,
        ),
        (
            "categorical",
            categorical_pipeline,
            nominal_cols,
        ),
    ])

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    pipeline = Pipeline([
        (
            "preprocess",
            preprocessor,
        ),
        (
            "model",
            random_forest,
        ),
    ])

    return pipeline


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(
        "Loading data...",
        flush=True,
    )

    df = load_data()

    (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols,
    ) = prepare_data(df)

    # The held-out evaluation sample is deliberately unused.
    del X_eval
    del y_eval

    print(
        f"Tuning observations: {len(X_tune):,}",
        flush=True,
    )

    print(
        "Held-out evaluation sample is not used.",
        flush=True,
    )

    # -------------------------------------------------------------
    # Fixed folds
    #
    # Every tree-count configuration is evaluated on the same
    # partitions so that the comparison is paired.
    # -------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            X_tune,
            y_tune,
        )
    )

    results = []

    # -------------------------------------------------------------
    # Question
    #
    # Does increasing the forest from 100 to 200 or 400 trees
    # materially improve validation Macro F1 or stability once the
    # selected Random Forest settings are used?
    # -------------------------------------------------------------

    for n_estimators in TREE_COUNTS:
        print(
            f"\nTesting n_estimators = {n_estimators}",
            flush=True,
        )

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splits,
            start=1,
        ):
            print(
                f"  Fold {fold_number}/{N_SPLITS}...",
                flush=True,
            )

            X_train = X_tune.iloc[
                train_index
            ]

            X_validation = X_tune.iloc[
                validation_index
            ]

            y_train = y_tune.iloc[
                train_index
            ]

            y_validation = y_tune.iloc[
                validation_index
            ]

            model = build_pipeline(
                numeric_cols=numeric_cols,
                nominal_cols=nominal_cols,
                n_estimators=n_estimators,
            )

            start_time = perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            fit_seconds = (
                perf_counter()
                - start_time
            )

            train_predictions = model.predict(
                X_train
            )

            validation_predictions = model.predict(
                X_validation
            )

            train_macro_f1 = f1_score(
                y_train,
                train_predictions,
                average="macro",
                zero_division=0,
            )

            validation_macro_f1 = f1_score(
                y_validation,
                validation_predictions,
                average="macro",
                zero_division=0,
            )

            validation_accuracy = accuracy_score(
                y_validation,
                validation_predictions,
            )

            results.append(
                {
                    "n_estimators":
                        n_estimators,
                    "fold":
                        fold_number,
                    "train_macro_f1":
                        train_macro_f1,
                    "validation_macro_f1":
                        validation_macro_f1,
                    "validation_accuracy":
                        validation_accuracy,
                    "fit_seconds":
                        fit_seconds,
                }
            )

            print(
                f"    Train Macro F1: "
                f"{train_macro_f1:.4f}",
                flush=True,
            )

            print(
                f"    Validation Macro F1: "
                f"{validation_macro_f1:.4f}",
                flush=True,
            )

            print(
                f"    Validation accuracy: "
                f"{validation_accuracy:.4f}",
                flush=True,
            )

            print(
                f"    Fit time: "
                f"{fit_seconds:.1f}s",
                flush=True,
            )

    # -------------------------------------------------------------
    # Save fold-level results
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    folds_path = (
        RESULTS_DIR
        / "rf_tree_count_confirm_folds.csv"
    )

    results_df.to_csv(
        folds_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Summarise each tree count
    # -------------------------------------------------------------

    summary_df = (
        results_df
        .groupby(
            "n_estimators",
            as_index=False,
        )
        .agg(
            train_macro_f1_mean=(
                "train_macro_f1",
                "mean",
            ),
            train_macro_f1_std=(
                "train_macro_f1",
                "std",
            ),
            validation_macro_f1_mean=(
                "validation_macro_f1",
                "mean",
            ),
            validation_macro_f1_std=(
                "validation_macro_f1",
                "std",
            ),
            validation_accuracy_mean=(
                "validation_accuracy",
                "mean",
            ),
            validation_accuracy_std=(
                "validation_accuracy",
                "std",
            ),
            fit_seconds_mean=(
                "fit_seconds",
                "mean",
            ),
        )
    )

    # Difference relative to the 100-tree configuration.
    baseline_macro_f1 = summary_df.loc[
        summary_df["n_estimators"] == 100,
        "validation_macro_f1_mean",
    ].iloc[0]

    summary_df[
        "macro_f1_change_vs_100"
    ] = (
        summary_df[
            "validation_macro_f1_mean"
        ]
        - baseline_macro_f1
    )

    summary_path = (
        RESULTS_DIR
        / "rf_tree_count_confirm_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------

    print(
        "\n========================================",
        flush=True,
    )

    print(
        "TREE-COUNT CONFIRMATION COMPLETE",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    print(
        "\nSummary:",
        flush=True,
    )

    print(
        summary_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        f"\nFold-level results saved to: "
        f"{folds_path}",
        flush=True,
    )

    print(
        f"Summary saved to: "
        f"{summary_path}",
        flush=True,
    )

    print(
        "\nNo tree count is selected automatically.",
        flush=True,
    )

    print(
        "The decision should consider mean Macro F1, "
        "fold-to-fold variability and computational cost.",
        flush=True,
    )


if __name__ == "__main__":
    main()