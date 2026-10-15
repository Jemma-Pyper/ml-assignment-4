from pathlib import Path
import pandas as pd
from data import load_data, prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

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
    # Combine predictors and target
    # -------------------------------------------------------------

    tune_df = X_tune.copy()

    tune_df["target"] = (
        y_tune
        .reset_index(drop=True)
        .values
    )

    # -------------------------------------------------------------
    # Identify duplicated predictor patterns
    #
    # keep=False marks every row belonging to a duplicated predictor
    # pattern, including the first occurrence.
    # -------------------------------------------------------------

    duplicated_predictor_mask = (
        X_tune.duplicated(
            keep=False
        )
    )

    duplicated_predictor_rows = (
        tune_df.loc[
            duplicated_predictor_mask
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Group identical predictor rows and count distinct labels
    # -------------------------------------------------------------

    predictor_columns = list(
        X_tune.columns
    )

    grouped = (
        duplicated_predictor_rows
        .groupby(
            predictor_columns,
            dropna=False,
        )
        .agg(
            row_count=(
                "target",
                "size",
            ),
            distinct_target_count=(
                "target",
                "nunique",
            ),
            targets=(
                "target",
                lambda values:
                    ",".join(
                        str(value)
                        for value
                        in sorted(
                            set(values)
                        )
                    ),
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------
    # Conflicting predictor patterns
    #
    # A conflict exists when identical predictor values appear with
    # more than one target label.
    # -------------------------------------------------------------

    conflicting_groups = (
        grouped[
            grouped[
                "distinct_target_count"
            ] > 1
        ]
        .copy()
    )

    consistent_duplicate_groups = (
        grouped[
            grouped[
                "distinct_target_count"
            ] == 1
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Summary counts
    # -------------------------------------------------------------

    duplicate_pattern_count = len(
        grouped
    )

    conflicting_pattern_count = len(
        conflicting_groups
    )

    consistent_pattern_count = len(
        consistent_duplicate_groups
    )

    duplicated_rows_total = int(
        duplicated_predictor_mask.sum()
    )

    conflicting_row_count = int(
        conflicting_groups[
            "row_count"
        ].sum()
    ) if not conflicting_groups.empty else 0

    consistent_duplicate_row_count = int(
        consistent_duplicate_groups[
            "row_count"
        ].sum()
    ) if not consistent_duplicate_groups.empty else 0

    conflict_pattern_percentage = (
        100
        * conflicting_pattern_count
        / duplicate_pattern_count
        if duplicate_pattern_count > 0
        else 0.0
    )

    conflict_row_percentage = (
        100
        * conflicting_row_count
        / duplicated_rows_total
        if duplicated_rows_total > 0
        else 0.0
    )

    summary_df = pd.DataFrame(
        [
            {
                "measure":
                    "tuning_observations",
                "value":
                    len(X_tune),
            },
            {
                "measure":
                    "duplicated_predictor_rows_all_occurrences",
                "value":
                    duplicated_rows_total,
            },
            {
                "measure":
                    "duplicate_predictor_patterns",
                "value":
                    duplicate_pattern_count,
            },
            {
                "measure":
                    "consistent_duplicate_patterns",
                "value":
                    consistent_pattern_count,
            },
            {
                "measure":
                    "conflicting_duplicate_patterns",
                "value":
                    conflicting_pattern_count,
            },
            {
                "measure":
                    "rows_in_consistent_duplicate_patterns",
                "value":
                    consistent_duplicate_row_count,
            },
            {
                "measure":
                    "rows_in_conflicting_duplicate_patterns",
                "value":
                    conflicting_row_count,
            },
            {
                "measure":
                    "conflicting_pattern_percentage",
                "value":
                    conflict_pattern_percentage,
            },
            {
                "measure":
                    "conflicting_row_percentage",
                "value":
                    conflict_row_percentage,
            },
        ]
    )

    # -------------------------------------------------------------
    # Optional summary of target combinations in conflicting groups
    # -------------------------------------------------------------

    if not conflicting_groups.empty:
        target_combination_df = (
            conflicting_groups[
                "targets"
            ]
            .value_counts()
            .rename_axis(
                "target_combination"
            )
            .reset_index(
                name="pattern_count"
            )
        )
    else:
        target_combination_df = pd.DataFrame(
            columns=[
                "target_combination",
                "pattern_count",
            ]
        )

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "duplicate_label_summary.csv"
    )

    conflicts_path = (
        RESULTS_DIR
        / "duplicate_label_conflicts.csv"
    )

    combinations_path = (
        RESULTS_DIR
        / "duplicate_label_target_combinations.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    conflicting_groups.to_csv(
        conflicts_path,
        index=False,
    )

    target_combination_df.to_csv(
        combinations_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Terminal output
    # -------------------------------------------------------------

    print(
        "\n========================================",
        flush=True,
    )

    print(
        "DUPLICATE-LABEL CHECK COMPLETE",
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

    if conflicting_groups.empty:
        print(
            "\nNo identical predictor patterns "
            "with conflicting target labels were found.",
            flush=True,
        )
    else:
        print(
            "\nConflicting duplicate patterns were found.",
            flush=True,
        )

        print(
            f"Number of conflicting patterns: "
            f"{conflicting_pattern_count:,}",
            flush=True,
        )

        print(
            f"Rows belonging to conflicting patterns: "
            f"{conflicting_row_count:,}",
            flush=True,
        )

        print(
            "\nTarget-label combinations:",
            flush=True,
        )

        print(
            target_combination_df.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        f"\nSummary saved to: "
        f"{summary_path}",
        flush=True,
    )

    print(
        f"Conflicting patterns saved to: "
        f"{conflicts_path}",
        flush=True,
    )

    print(
        f"Target combinations saved to: "
        f"{combinations_path}",
        flush=True,
    )

    print(
        "\nNo observations were removed.",
        flush=True,
    )

    print(
        "No preprocessing or model settings were changed.",
        flush=True,
    )

if __name__ == "__main__":
    main()