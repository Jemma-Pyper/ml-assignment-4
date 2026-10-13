from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

MAX_FEATURES_VALUES = [
    "log2",
    "sqrt",
    0.5,
    0.75,
    None
]


def build_pipeline(
    numeric_cols,
    nominal_cols,
    max_features
):
    """Create the Random Forest pipeline for max_features tuning."""

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
        min_samples_leaf=5,
        max_features=max_features,
        bootstrap=True,
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

    # Final evaluation data is deliberately not used during tuning.
    del X_eval, y_eval

    tuning_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for max_features in MAX_FEATURES_VALUES:
        print(f"\nTesting max_features = {max_features}...")

        model = build_pipeline(
            numeric_cols=numeric_cols,
            nominal_cols=nominal_cols,
            max_features=max_features
        )

        scores = cross_validate(
            model,
            X_tune,
            y_tune,
            cv=tuning_cv,
            scoring={
                "f1_macro": "f1_macro",
                "accuracy": "accuracy"
            },
            return_train_score=True,
            n_jobs=1
        )

        result = {
            "max_features": str(max_features),
            "train_macro_f1_mean":
                scores["train_f1_macro"].mean(),
            "train_macro_f1_std":
                scores["train_f1_macro"].std(),
            "validation_macro_f1_mean":
                scores["test_f1_macro"].mean(),
            "validation_macro_f1_std":
                scores["test_f1_macro"].std(),
            "validation_accuracy_mean":
                scores["test_accuracy"].mean(),
            "validation_accuracy_std":
                scores["test_accuracy"].std()
        }

        results.append(result)

        print(
            "Training Macro F1:",
            f"{result['train_macro_f1_mean']:.4f}",
            "+/-",
            f"{result['train_macro_f1_std']:.4f}"
        )

        print(
            "Validation Macro F1:",
            f"{result['validation_macro_f1_mean']:.4f}",
            "+/-",
            f"{result['validation_macro_f1_std']:.4f}"
        )

        print(
            "Validation Accuracy:",
            f"{result['validation_accuracy_mean']:.4f}",
            "+/-",
            f"{result['validation_accuracy_std']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "rf_max_features.csv"
    results_df.to_csv(output_path, index=False)

    print("\nSummary:")
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()