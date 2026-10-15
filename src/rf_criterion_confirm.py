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

CRITERIA = [
    "entropy",
    "gini",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_pipeline(
    numeric_cols,
    nominal_cols,
    criterion,
):
    """
    Construct the current Random Forest candidate while varying only
    the splitting criterion.
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
        n_estimators=200,
        criterion=criterion,
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

    # The held-out evaluation sample remains unused.
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
    # Fixed stratified folds
    #
    # Both splitting criteria use exactly the same partitions.
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
    # Does the entropy splitting criterion inherited from the
    # Assignment 2 classification tree remain preferable for the
    # tuned Random Forest, or does Gini impurity provide better
    # validation performance?
    # -------------------------------------------------------------

    for criterion in CRITERIA:
        print(
            f"\nTesting criterion = {criterion}",
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
                criterion=criterion,
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
                    "criterion":
                        criterion,
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
        / "rf_criterion_confirm_folds.csv"
    )

    results_df.to_csv(
        folds_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Summarise each splitting criterion
    # -------------------------------------------------------------

    summary_df = (
        results_df
        .groupby(
            "criterion",
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

    entropy_macro_f1 = summary_df.loc[
        summary_df["criterion"] == "entropy",
        "validation_macro_f1_mean",
    ].iloc[0]

    summary_df[
        "macro_f1_change_vs_entropy"
    ] = (
        summary_df[
            "validation_macro_f1_mean"
        ]
        - entropy_macro_f1
    )

    summary_path = (
        RESULTS_DIR
        / "rf_criterion_confirm_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Paired fold comparison
    # -------------------------------------------------------------

    entropy_results = (
        results_df[
            results_df["criterion"] == "entropy"
        ]
        .set_index("fold")
    )

    gini_results = (
        results_df[
            results_df["criterion"] == "gini"
        ]
        .set_index("fold")
    )

    paired_df = pd.DataFrame({
        "fold":
            entropy_results.index,
        "entropy_macro_f1":
            entropy_results[
                "validation_macro_f1"
            ].values,
        "gini_macro_f1":
            gini_results[
                "validation_macro_f1"
            ].values,
    })

    paired_df[
        "gini_minus_entropy"
    ] = (
        paired_df["gini_macro_f1"]
        - paired_df["entropy_macro_f1"]
    )

    paired_path = (
        RESULTS_DIR
        / "rf_criterion_confirm_paired.csv"
    )

    paired_df.to_csv(
        paired_path,
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
        "CRITERION CONFIRMATION COMPLETE",
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
        "\nPaired fold comparison:",
        flush=True,
    )

    print(
        paired_df.to_string(
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
        f"Paired comparison saved to: "
        f"{paired_path}",
        flush=True,
    )

    print(
        "\nNo splitting criterion is selected automatically.",
        flush=True,
    )

    print(
        "Selection should consider validation Macro F1, "
        "fold-to-fold consistency and computational cost.",
        flush=True,
    )


if __name__ == "__main__":
    main()