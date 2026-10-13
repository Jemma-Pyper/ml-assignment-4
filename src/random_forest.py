from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


RANDOM_STATE = 42


def build_random_forest_pipeline(numeric_cols, nominal_cols):
    """Create the preprocessing and initial Random Forest pipeline."""

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
        max_features="sqrt",
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
    df = load_data()

    (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols
    ) = prepare_data(df)

    # X_eval and y_eval are deliberately not used during tuning.
    del X_eval, y_eval

    model = build_random_forest_pipeline(
        numeric_cols,
        nominal_cols
    )

    tuning_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = cross_validate(
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

    print("Initial Random Forest")

    print(
        "Training Macro F1:",
        f"{results['train_f1_macro'].mean():.4f}",
        "+/-",
        f"{results['train_f1_macro'].std():.4f}"
    )

    print(
        "Validation Macro F1:",
        f"{results['test_f1_macro'].mean():.4f}",
        "+/-",
        f"{results['test_f1_macro'].std():.4f}"
    )

    print(
        "Validation Accuracy:",
        f"{results['test_accuracy'].mean():.4f}",
        "+/-",
        f"{results['test_accuracy'].std():.4f}"
    )


if __name__ == "__main__":
    main()