from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

N_SPLITS = 3
N_REPEATS = 5

CLASS_WEIGHT_VALUES = [
    ("none", None),
    ("balanced", "balanced")
]


def build_pipeline(
    numeric_cols,
    nominal_cols,
    class_weight
):
    """Create the Random Forest pipeline for class-weight confirmation."""

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            nominal_cols
        )
    ])

    random_forest = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=2
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", random_forest)
    ])

    return pipeline


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data()

    (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols
    ) = prepare_data(df)

    # The final evaluation sample remains untouched.
    del X_eval, y_eval

    repeated_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    # Create the splits once so that both class-weight settings
    # are evaluated on exactly the same partitions.
    splits = list(
        repeated_cv.split(
            X_tune,
            y_tune
        )
    )

    results = []

    for weight_name, class_weight in CLASS_WEIGHT_VALUES:
        print(
            f"\nTesting class_weight = {weight_name}"
        )

        for split_number, (train_index, validation_index) in enumerate(
            splits
        ):
            repeat_number = (
                split_number // N_SPLITS
            ) + 1

            fold_number = (
                split_number % N_SPLITS
            ) + 1

            print(
                f"Repeat {repeat_number}/{N_REPEATS}, "
                f"fold {fold_number}/{N_SPLITS}..."
            )

            X_train = X_tune.iloc[train_index]
            X_validation = X_tune.iloc[validation_index]

            y_train = y_tune.iloc[train_index]
            y_validation = y_tune.iloc[validation_index]

            model = build_pipeline(
                numeric_cols=numeric_cols,
                nominal_cols=nominal_cols,
                class_weight=class_weight
            )

            model.fit(
                X_train,
                y_train
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
                zero_division=0
            )

            validation_macro_f1 = f1_score(
                y_validation,
                validation_predictions,
                average="macro",
                zero_division=0
            )

            validation_accuracy = accuracy_score(
                y_validation,
                validation_predictions
            )

            results.append({
                "class_weight": weight_name,
                "repeat": repeat_number,
                "fold": fold_number,
                "train_macro_f1": train_macro_f1,
                "validation_macro_f1": validation_macro_f1,
                "validation_accuracy": validation_accuracy
            })

            print(
                "  Validation Macro F1:",
                f"{validation_macro_f1:.4f}"
            )

    results_df = pd.DataFrame(
        results
    )

    folds_path = (
        RESULTS_DIR
        / "rf_class_weight_confirm_folds.csv"
    )

    results_df.to_csv(
        folds_path,
        index=False
    )

    # Summarise each class-weight setting across all 15 folds.
    summary_rows = []

    for weight_name, _ in CLASS_WEIGHT_VALUES:
        subset = results_df[
            results_df["class_weight"] == weight_name
        ]

        summary_rows.append({
            "class_weight": weight_name,
            "train_macro_f1_mean":
                subset["train_macro_f1"].mean(),
            "train_macro_f1_std":
                subset["train_macro_f1"].std(ddof=0),
            "validation_macro_f1_mean":
                subset["validation_macro_f1"].mean(),
            "validation_macro_f1_std":
                subset["validation_macro_f1"].std(ddof=0),
            "validation_accuracy_mean":
                subset["validation_accuracy"].mean(),
            "validation_accuracy_std":
                subset["validation_accuracy"].std(ddof=0)
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "rf_class_weight_confirm_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # Create a paired comparison because both models used
    # exactly the same repeated cross-validation partitions.
    none_results = results_df[
        results_df["class_weight"] == "none"
    ][
        [
            "repeat",
            "fold",
            "validation_macro_f1",
            "validation_accuracy"
        ]
    ].copy()

    balanced_results = results_df[
        results_df["class_weight"] == "balanced"
    ][
        [
            "repeat",
            "fold",
            "validation_macro_f1",
            "validation_accuracy"
        ]
    ].copy()

    comparison_df = none_results.merge(
        balanced_results,
        on=[
            "repeat",
            "fold"
        ],
        suffixes=(
            "_none",
            "_balanced"
        )
    )

    comparison_df[
        "macro_f1_change_balanced_minus_none"
    ] = (
        comparison_df[
            "validation_macro_f1_balanced"
        ]
        - comparison_df[
            "validation_macro_f1_none"
        ]
    )

    comparison_df[
        "accuracy_change_balanced_minus_none"
    ] = (
        comparison_df[
            "validation_accuracy_balanced"
        ]
        - comparison_df[
            "validation_accuracy_none"
        ]
    )

    comparison_path = (
        RESULTS_DIR
        / "rf_class_weight_confirm_paired.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # Summarise the paired comparison within each repeat.
    repeat_summary = (
        comparison_df
        .groupby(
            "repeat",
            as_index=False
        )
        .agg(
            validation_macro_f1_none=(
                "validation_macro_f1_none",
                "mean"
            ),
            validation_macro_f1_balanced=(
                "validation_macro_f1_balanced",
                "mean"
            ),
            macro_f1_change=(
                "macro_f1_change_balanced_minus_none",
                "mean"
            ),
            validation_accuracy_none=(
                "validation_accuracy_none",
                "mean"
            ),
            validation_accuracy_balanced=(
                "validation_accuracy_balanced",
                "mean"
            ),
            accuracy_change=(
                "accuracy_change_balanced_minus_none",
                "mean"
            )
        )
    )

    repeat_path = (
        RESULTS_DIR
        / "rf_class_weight_confirm_repeats.csv"
    )

    repeat_summary.to_csv(
        repeat_path,
        index=False
    )

    balanced_wins = (
        comparison_df[
            "macro_f1_change_balanced_minus_none"
        ] > 0
    ).sum()

    mean_macro_f1_change = comparison_df[
        "macro_f1_change_balanced_minus_none"
    ].mean()

    print("\nOverall summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nMean results by repeat:")
    print(
        repeat_summary.to_string(
            index=False
        )
    )

    print(
        f"\nBalanced weighting produced the higher "
        f"Macro F1 in {balanced_wins} of "
        f"{len(comparison_df)} validation folds."
    )

    print(
        "Mean paired Macro F1 change "
        "(balanced - none):",
        f"{mean_macro_f1_change:.4f}"
    )

    print(
        f"\nFold results saved to: {folds_path}"
    )

    print(
        f"Summary saved to: {summary_path}"
    )

    print(
        f"Paired comparison saved to: "
        f"{comparison_path}"
    )

    print(
        f"Repeat summary saved to: "
        f"{repeat_path}"
    )


if __name__ == "__main__":
    main()