from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import load_data, prepare_data


RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Analysis",
    6: "Fuzzers",
    7: "Worms",
    8: "Shellcode",
    9: "Generic",
}

CLASS_LABELS = list(CLASS_NAMES.keys())


def categorical_pipeline():
    return Pipeline([
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


def build_knn(numeric_cols, nominal_cols):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            MinMaxScaler(),
            numeric_cols,
        ),
        (
            "categorical",
            categorical_pipeline(),
            nominal_cols,
        ),
    ])

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="manhattan",
        n_jobs=N_JOBS,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def build_decision_tree(numeric_cols, nominal_cols):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric_cols,
        ),
        (
            "categorical",
            categorical_pipeline(),
            nominal_cols,
        ),
    ])

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def build_random_forest(numeric_cols, nominal_cols):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            numeric_cols,
        ),
        (
            "categorical",
            categorical_pipeline(),
            nominal_cols,
        ),
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate_model(
    model_name,
    model_builder,
    X_eval,
    y_eval,
    numeric_cols,
    nominal_cols,
    splits,
):
    print("\n========================================", flush=True)
    print(f"EVALUATING: {model_name}", flush=True)
    print("========================================", flush=True)

    fold_results = []
    oof_predictions = np.empty(
        len(y_eval),
        dtype=y_eval.dtype,
    )

    for fold_number, (train_index, validation_index) in enumerate(
        splits,
        start=1,
    ):
        print(f"\nFold {fold_number}/{N_SPLITS}...", flush=True)

        X_train = X_eval.iloc[train_index]
        X_validation = X_eval.iloc[validation_index]
        y_train = y_eval.iloc[train_index]
        y_validation = y_eval.iloc[validation_index]

        model = model_builder(
            numeric_cols,
            nominal_cols,
        )

        fit_start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = model.predict(X_validation)
        predict_seconds = perf_counter() - predict_start

        oof_predictions[validation_index] = predictions

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        fold_results.append({
            "model": model_name,
            "fold": fold_number,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        })

        print(f"  Macro F1: {macro_f1:.4f}", flush=True)
        print(f"  Accuracy: {accuracy:.4f}", flush=True)
        print(f"  Fit time: {fit_seconds:.1f}s", flush=True)
        print(
            f"  Prediction time: {predict_seconds:.1f}s",
            flush=True,
        )

    return pd.DataFrame(fold_results), oof_predictions


def make_class_report(
    model_name,
    y_true,
    predictions,
):
    report = classification_report(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        target_names=[
            CLASS_NAMES[label]
            for label in CLASS_LABELS
        ],
        output_dict=True,
        zero_division=0,
    )

    if not isinstance(report, dict):
        raise TypeError(
            "classification_report did not return a dictionary."
        )

    rows = []

    for label in CLASS_LABELS:
        class_name = CLASS_NAMES[label]
        values = report[class_name]

        if not isinstance(values, dict):
            raise TypeError(
                f"Unexpected classification-report entry for {class_name}."
            )

        rows.append({
            "model": model_name,
            "class_id": label,
            "attack_category": class_name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "support": int(values["support"]),
        })

    return pd.DataFrame(rows)


def make_confusion_matrices(
    y_true,
    predictions,
):
    raw = confusion_matrix(
        y_true,
        predictions,
        labels=CLASS_LABELS,
    )

    normalised = confusion_matrix(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        normalize="true",
    )

    names = [
        CLASS_NAMES[label]
        for label in CLASS_LABELS
    ]

    raw_df = pd.DataFrame(
        raw,
        index=names,
        columns=names,
    )

    normalised_df = pd.DataFrame(
        normalised,
        index=names,
        columns=names,
    )

    raw_df.index.name = "actual"
    raw_df.columns.name = "predicted"

    normalised_df.index.name = "actual"
    normalised_df.columns.name = "predicted"

    return raw_df, normalised_df


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...", flush=True)

    df = load_data()

    (
        X_tune,
        X_eval,
        y_tune,
        y_eval,
        numeric_cols,
        nominal_cols,
    ) = prepare_data(df)

    del X_tune
    del y_tune

    print("\nFINAL EVALUATION", flush=True)
    print(
        "The held-out evaluation sample is now being accessed.",
        flush=True,
    )
    print(
        f"Evaluation observations: {len(X_eval):,}",
        flush=True,
    )
    print(
        f"Predictors: {X_eval.shape[1]}",
        flush=True,
    )
    print("\nIMPORTANT:", flush=True)
    print(
        "Model and preprocessing settings are frozen.",
        flush=True,
    )
    print(
        "Final evaluation results must not be used for further tuning.",
        flush=True,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            X_eval,
            y_eval,
        )
    )

    models = [
        ("kNN", build_knn),
        ("Decision Tree", build_decision_tree),
        ("Random Forest", build_random_forest),
    ]

    all_fold_results = []
    all_class_reports = []

    all_predictions = {
        "true_label": y_eval.to_numpy(),
    }

    random_forest_predictions = None

    for model_name, model_builder in models:
        fold_results, predictions = evaluate_model(
            model_name=model_name,
            model_builder=model_builder,
            X_eval=X_eval,
            y_eval=y_eval,
            numeric_cols=numeric_cols,
            nominal_cols=nominal_cols,
            splits=splits,
        )

        all_fold_results.append(
            fold_results
        )

        class_report = make_class_report(
            model_name=model_name,
            y_true=y_eval,
            predictions=predictions,
        )

        all_class_reports.append(
            class_report
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        all_predictions[
            f"{safe_name}_prediction"
        ] = predictions

        if model_name == "Random Forest":
            random_forest_predictions = predictions.copy()

    fold_results_df = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    fold_results_path = (
        RESULTS_DIR
        / "final_evaluation_folds.csv"
    )

    fold_results_df.to_csv(
        fold_results_path,
        index=False,
    )

    summary_df = (
        fold_results_df
        .groupby(
            "model",
            as_index=False,
            sort=False,
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            fit_seconds_mean=(
                "fit_seconds",
                "mean",
            ),
            predict_seconds_mean=(
                "predict_seconds",
                "mean",
            ),
        )
    )

    summary_path = (
        RESULTS_DIR
        / "final_evaluation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    pivot_macro = (
        fold_results_df
        .pivot(
            index="fold",
            columns="model",
            values="macro_f1",
        )
        .reset_index()
    )

    paired_df = pd.DataFrame({
        "fold":
            pivot_macro["fold"],
        "knn_macro_f1":
            pivot_macro["kNN"],
        "decision_tree_macro_f1":
            pivot_macro["Decision Tree"],
        "random_forest_macro_f1":
            pivot_macro["Random Forest"],
    })

    paired_df[
        "rf_minus_knn"
    ] = (
        paired_df[
            "random_forest_macro_f1"
        ]
        - paired_df[
            "knn_macro_f1"
        ]
    )

    paired_df[
        "rf_minus_decision_tree"
    ] = (
        paired_df[
            "random_forest_macro_f1"
        ]
        - paired_df[
            "decision_tree_macro_f1"
        ]
    )

    paired_path = (
        RESULTS_DIR
        / "final_evaluation_paired.csv"
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    class_reports_df = pd.concat(
        all_class_reports,
        ignore_index=True,
    )

    class_reports_path = (
        RESULTS_DIR
        / "final_evaluation_class_reports.csv"
    )

    class_reports_df.to_csv(
        class_reports_path,
        index=False,
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    predictions_path = (
        RESULTS_DIR
        / "final_evaluation_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    if random_forest_predictions is None:
        raise RuntimeError(
            "Random Forest predictions were not generated."
        )

    (
        rf_confusion_raw,
        rf_confusion_normalised,
    ) = make_confusion_matrices(
        y_true=y_eval,
        predictions=random_forest_predictions,
    )

    raw_confusion_path = (
        RESULTS_DIR
        / "final_rf_confusion_raw.csv"
    )

    normalised_confusion_path = (
        RESULTS_DIR
        / "final_rf_confusion_normalised.csv"
    )

    rf_confusion_raw.to_csv(
        raw_confusion_path
    )

    rf_confusion_normalised.to_csv(
        normalised_confusion_path
    )

    print("\n========================================", flush=True)
    print("FINAL EVALUATION COMPLETE", flush=True)
    print("========================================", flush=True)

    print("\nOverall summary:", flush=True)
    print(
        summary_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nPaired fold Macro F1 comparison:",
        flush=True,
    )

    print(
        paired_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRandom Forest class-specific results:",
        flush=True,
    )

    print(
        class_reports_df[
            class_reports_df[
                "model"
            ] == "Random Forest"
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nRandom Forest normalised confusion matrix:",
        flush=True,
    )

    print(
        rf_confusion_normalised
        .round(4)
        .to_string(),
        flush=True,
    )

    print("\nResults saved to:", flush=True)
    print(fold_results_path, flush=True)
    print(summary_path, flush=True)
    print(paired_path, flush=True)
    print(class_reports_path, flush=True)
    print(predictions_path, flush=True)
    print(raw_confusion_path, flush=True)
    print(normalised_confusion_path, flush=True)

    print(
        "\nFINAL MODEL SETTINGS ARE NOW LOCKED.",
        flush=True,
    )
    print(
        "Do not modify the models based on these evaluation results.",
        flush=True,
    )

if __name__ == "__main__":
    main()