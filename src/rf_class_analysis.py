from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_WEIGHT_VALUES = [
    None,
    "balanced"
]


def build_pipeline(
    numeric_cols,
    nominal_cols,
    class_weight
):
    """Create the Random Forest pipeline for class-level analysis."""

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

    tuning_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_reports = []

    for class_weight in CLASS_WEIGHT_VALUES:
        print(
            f"\nGenerating out-of-fold predictions for "
            f"class_weight = {class_weight}..."
        )

        model = build_pipeline(
            numeric_cols=numeric_cols,
            nominal_cols=nominal_cols,
            class_weight=class_weight
        )

        predictions = cross_val_predict(
            model,
            X_tune,
            y_tune,
            cv=tuning_cv,
            method="predict",
            n_jobs=1
        )

        macro_f1 = f1_score(
            y_tune,
            predictions,
            average="macro"
        )

        accuracy = accuracy_score(
            y_tune,
            predictions
        )

        print(
            f"\nOverall Macro F1: {macro_f1:.4f}"
        )
        print(
            f"Overall Accuracy: {accuracy:.4f}"
        )

        report = classification_report(
            y_tune,
            predictions,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        report_df["class_weight"] = str(
            class_weight
        )

        all_reports.append(report_df)

        print("\nClassification report:")
        print(
            classification_report(
                y_tune,
                predictions,
                digits=4,
                zero_division=0
            )
        )

        safe_name = (
            "none"
            if class_weight is None
            else str(class_weight)
        )

        output_path = (
            RESULTS_DIR
            / f"rf_class_report_{safe_name}.csv"
        )

        report_df.to_csv(
            output_path,
            index=True
        )

        print(
            f"Class report saved to: {output_path}"
        )

    combined_df = pd.concat(
        all_reports,
        keys=[
            "None",
            "balanced"
        ],
        names=[
            "weight_setting",
            "class"
        ]
    )

    combined_path = (
        RESULTS_DIR
        / "rf_class_analysis.csv"
    )

    combined_df.to_csv(
        combined_path
    )

    print(
        f"\nCombined results saved to: "
        f"{combined_path}"
    )

    # Create a direct per-class comparison.
    none_report = all_reports[0]
    balanced_report = all_reports[1]

    class_labels = sorted(
        y_tune.unique()
    )

    comparison_rows = []

    for class_label in class_labels:
        none_f1 = none_report.loc[
            str(class_label),
            "f1-score"
        ]

        balanced_f1 = balanced_report.loc[
            str(class_label),
            "f1-score"
        ]

        comparison_rows.append({
            "class": class_label,
            "f1_none": none_f1,
            "f1_balanced": balanced_f1,
            "f1_change":
                balanced_f1 - none_f1,
            "precision_none":
                none_report.loc[
                    str(class_label),
                    "precision"
                ],
            "precision_balanced":
                balanced_report.loc[
                    str(class_label),
                    "precision"
                ],
            "recall_none":
                none_report.loc[
                    str(class_label),
                    "recall"
                ],
            "recall_balanced":
                balanced_report.loc[
                    str(class_label),
                    "recall"
                ],
            "support":
                none_report.loc[
                    str(class_label),
                    "support"
                ]
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = comparison_df.sort_values(
        by="f1_change",
        ascending=False
    )

    comparison_path = (
        RESULTS_DIR
        / "rf_class_weight_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        "\nPer-class F1 comparison:"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        f"\nComparison saved to: "
        f"{comparison_path}"
    )


if __name__ == "__main__":
    main()