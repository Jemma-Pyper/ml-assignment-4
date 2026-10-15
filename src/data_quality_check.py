from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from data import load_data, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CORRELATION_THRESHOLD = 0.90


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
        f"Predictors: {X_tune.shape[1]}",
        flush=True,
    )

    print(
        "Held-out evaluation sample is not used.",
        flush=True,
    )

    # -------------------------------------------------------------
    # Basic overview
    # -------------------------------------------------------------

    duplicate_predictor_rows = int(
        X_tune.duplicated().sum()
    )

    combined_tune = pd.concat(
        [
            X_tune.reset_index(drop=True),
            y_tune.reset_index(drop=True).rename(
                "target"
            ),
        ],
        axis=1,
    )

    duplicate_complete_rows = int(
        combined_tune.duplicated().sum()
    )

    constant_predictors = [
        column
        for column in X_tune.columns
        if X_tune[column].nunique(
            dropna=False
        ) <= 1
    ]

    overview_rows = [
        {
            "measure":
                "tuning_observations",
            "value":
                len(X_tune),
        },
        {
            "measure":
                "predictor_count",
            "value":
                X_tune.shape[1],
        },
        {
            "measure":
                "numeric_predictor_count",
            "value":
                len(numeric_cols),
        },
        {
            "measure":
                "nominal_predictor_count",
            "value":
                len(nominal_cols),
        },
        {
            "measure":
                "duplicate_predictor_rows",
            "value":
                duplicate_predictor_rows,
        },
        {
            "measure":
                "duplicate_complete_rows",
            "value":
                duplicate_complete_rows,
        },
        {
            "measure":
                "constant_predictor_count",
            "value":
                len(constant_predictors),
        },
    ]

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    missing_rows = []

    for column in X_tune.columns:
        missing_count = int(
            X_tune[column].isna().sum()
        )

        missing_rows.append(
            {
                "predictor":
                    column,
                "missing_count":
                    missing_count,
                "missing_percentage":
                    100
                    * missing_count
                    / len(X_tune),
            }
        )

    missing_df = pd.DataFrame(
        missing_rows
    ).sort_values(
        "missing_percentage",
        ascending=False,
    )

    # -------------------------------------------------------------
    # Nominal predictors
    # -------------------------------------------------------------

    categorical_rows = []

    for column in nominal_cols:
        missing_count = int(
            X_tune[column].isna().sum()
        )

        unique_non_missing = int(
            X_tune[column].nunique(
                dropna=True
            )
        )

        categories_after_missing = (
            unique_non_missing
            + int(missing_count > 0)
        )

        categorical_rows.append(
            {
                "predictor":
                    column,
                "unique_non_missing":
                    unique_non_missing,
                "missing_count":
                    missing_count,
                "missing_percentage":
                    100
                    * missing_count
                    / len(X_tune),
                "categories_after_missing":
                    categories_after_missing,
            }
        )

    categorical_df = pd.DataFrame(
        categorical_rows
    )

    # -------------------------------------------------------------
    # One-hot encoded feature count
    #
    # This verifies how much the categorical representation expands
    # the Random Forest input space.
    # -------------------------------------------------------------

    categorical_for_encoding = (
        X_tune[nominal_cols]
        .fillna("Missing")
        .astype(str)
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    encoder.fit(
        categorical_for_encoding
    )

    encoded_feature_count = len(
        encoder.get_feature_names_out(
            nominal_cols
        )
    )

    transformed_predictor_count = (
        len(numeric_cols)
        + encoded_feature_count
    )

    overview_rows.extend(
        [
            {
                "measure":
                    "one_hot_encoded_predictor_count",
                "value":
                    encoded_feature_count,
            },
            {
                "measure":
                    "total_transformed_predictor_count",
                "value":
                    transformed_predictor_count,
            },
        ]
    )

    # -------------------------------------------------------------
    # Numeric predictor quality
    # -------------------------------------------------------------

    numeric_rows = []

    for column in numeric_cols:
        series = X_tune[column]

        finite_series = series[
            ~series.isin(
                [
                    float("inf"),
                    float("-inf"),
                ]
            )
        ].dropna()

        positive_infinity_count = int(
            (series == float("inf")).sum()
        )

        negative_infinity_count = int(
            (series == float("-inf")).sum()
        )

        negative_count = int(
            (finite_series < 0).sum()
        )

        zero_count = int(
            (finite_series == 0).sum()
        )

        if len(finite_series) > 0:
            q1 = finite_series.quantile(
                0.25
            )

            median = finite_series.median()

            q3 = finite_series.quantile(
                0.75
            )

            iqr = q3 - q1

            lower_bound = (
                q1
                - 1.5 * iqr
            )

            upper_bound = (
                q3
                + 1.5 * iqr
            )

            if iqr > 0:
                outlier_count = int(
                    (
                        (
                            finite_series
                            < lower_bound
                        )
                        |
                        (
                            finite_series
                            > upper_bound
                        )
                    ).sum()
                )
            else:
                outlier_count = 0

            outlier_percentage = (
                100
                * outlier_count
                / len(finite_series)
            )

            minimum = finite_series.min()
            maximum = finite_series.max()

        else:
            q1 = pd.NA
            median = pd.NA
            q3 = pd.NA
            iqr = pd.NA
            minimum = pd.NA
            maximum = pd.NA
            outlier_count = 0
            outlier_percentage = 0.0

        numeric_rows.append(
            {
                "predictor":
                    column,
                "minimum":
                    minimum,
                "q1":
                    q1,
                "median":
                    median,
                "q3":
                    q3,
                "maximum":
                    maximum,
                "iqr":
                    iqr,
                "negative_count":
                    negative_count,
                "zero_count":
                    zero_count,
                "positive_infinity_count":
                    positive_infinity_count,
                "negative_infinity_count":
                    negative_infinity_count,
                "iqr_outlier_count":
                    outlier_count,
                "iqr_outlier_percentage":
                    outlier_percentage,
            }
        )

    numeric_df = pd.DataFrame(
        numeric_rows
    ).sort_values(
        "iqr_outlier_percentage",
        ascending=False,
    )

    # -------------------------------------------------------------
    # Strong numeric correlations
    # -------------------------------------------------------------

    correlation_matrix = (
        X_tune[numeric_cols]
        .corr()
    )

    correlation_rows = []

    for column_a, column_b in combinations(
        numeric_cols,
        2,
    ):
        correlation = (
            correlation_matrix.loc[
                column_a,
                column_b,
            ]
        )

        if pd.isna(correlation):
            continue

        if (
            abs(correlation)
            >= CORRELATION_THRESHOLD
        ):
            correlation_rows.append(
                {
                    "predictor_1":
                        column_a,
                    "predictor_2":
                        column_b,
                    "correlation":
                        correlation,
                    "absolute_correlation":
                        abs(correlation),
                }
            )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    if not correlation_df.empty:
        correlation_df = (
            correlation_df
            .sort_values(
                "absolute_correlation",
                ascending=False,
            )
        )

    # -------------------------------------------------------------
    # Target class distribution
    # -------------------------------------------------------------

    class_counts = (
        y_tune
        .value_counts()
        .sort_index()
    )

    class_distribution_df = (
        class_counts
        .rename("count")
        .to_frame()
    )

    class_distribution_df[
        "percentage"
    ] = (
        100
        * class_distribution_df[
            "count"
        ]
        / len(y_tune)
    )

    class_distribution_df = (
        class_distribution_df
        .reset_index()
        .rename(
            columns={
                y_tune.name
                if y_tune.name
                is not None
                else "index":
                    "target",
            }
        )
    )

    if (
        "target"
        not in
        class_distribution_df.columns
    ):
        class_distribution_df = (
            class_distribution_df
            .rename(
                columns={
                    class_distribution_df
                    .columns[0]:
                        "target"
                }
            )
        )

    # -------------------------------------------------------------
    # Missing service by target category
    #
    # The check verifies whether missing service information has a
    # different prevalence across target categories.
    # -------------------------------------------------------------

    service_target_df = pd.DataFrame()

    if "service" in X_tune.columns:
        service_missing = (
            X_tune["service"]
            .isna()
            .rename(
                "service_missing"
            )
            .reset_index(
                drop=True
            )
        )

        target_reset = (
            y_tune
            .reset_index(
                drop=True
            )
            .rename(
                "target"
            )
        )

        service_analysis = pd.DataFrame(
            {
                "target":
                    target_reset,
                "service_missing":
                    service_missing,
            }
        )

        service_target_df = (
            service_analysis
            .groupby(
                "target",
                as_index=False,
            )
            .agg(
                observations=(
                    "service_missing",
                    "size",
                ),
                missing_service_count=(
                    "service_missing",
                    "sum",
                ),
                missing_service_rate=(
                    "service_missing",
                    "mean",
                ),
            )
        )

        service_target_df[
            "missing_service_percentage"
        ] = (
            100
            * service_target_df[
                "missing_service_rate"
            ]
        )

        service_target_df = (
            service_target_df.drop(
                columns=[
                    "missing_service_rate"
                ]
            )
        )

    # -------------------------------------------------------------
    # Predictor data types
    # -------------------------------------------------------------

    dtype_df = pd.DataFrame(
        {
            "predictor":
                X_tune.columns,
            "dtype":
                [
                    str(
                        X_tune[
                            column
                        ].dtype
                    )
                    for column
                    in X_tune.columns
                ],
            "model_role":
                [
                    (
                        "nominal"
                        if column
                        in nominal_cols
                        else "numeric"
                    )
                    for column
                    in X_tune.columns
                ],
        }
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    overview_df = pd.DataFrame(
        overview_rows
    )

    overview_df.to_csv(
        RESULTS_DIR
        / "data_quality_overview.csv",
        index=False,
    )

    missing_df.to_csv(
        RESULTS_DIR
        / "data_quality_missingness.csv",
        index=False,
    )

    categorical_df.to_csv(
        RESULTS_DIR
        / "data_quality_categorical.csv",
        index=False,
    )

    numeric_df.to_csv(
        RESULTS_DIR
        / "data_quality_numeric.csv",
        index=False,
    )

    correlation_df.to_csv(
        RESULTS_DIR
        / "data_quality_correlations.csv",
        index=False,
    )

    class_distribution_df.to_csv(
        RESULTS_DIR
        / "data_quality_class_distribution.csv",
        index=False,
    )

    dtype_df.to_csv(
        RESULTS_DIR
        / "data_quality_dtypes.csv",
        index=False,
    )

    if not service_target_df.empty:
        service_target_df.to_csv(
            RESULTS_DIR
            / "data_quality_service_by_class.csv",
            index=False,
        )

    # -------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------

    print(
        "\n========================================",
        flush=True,
    )

    print(
        "DATA-QUALITY CHECK COMPLETE",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    print(
        "\nOverview:",
        flush=True,
    )

    print(
        overview_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nMissing values:",
        flush=True,
    )

    print(
        missing_df[
            missing_df[
                "missing_count"
            ] > 0
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nNominal predictors:",
        flush=True,
    )

    print(
        categorical_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nTarget distribution:",
        flush=True,
    )

    print(
        class_distribution_df.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nStrong numeric correlations "
        f"(|r| >= {CORRELATION_THRESHOLD}):",
        flush=True,
    )

    if correlation_df.empty:
        print(
            "None found.",
            flush=True,
        )
    else:
        print(
            correlation_df.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        "\nNumeric predictors with "
        "the highest IQR outlier rates:",
        flush=True,
    )

    print(
        numeric_df[
            [
                "predictor",
                "minimum",
                "maximum",
                "negative_count",
                "positive_infinity_count",
                "negative_infinity_count",
                "iqr_outlier_percentage",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        ),
        flush=True,
    )

    if not service_target_df.empty:
        print(
            "\nMissing service by target:",
            flush=True,
        )

        print(
            service_target_df.to_string(
                index=False
            ),
            flush=True,
        )

    print(
        "\nNo data transformations or model "
        "parameters were changed by this script.",
        flush=True,
    )

    print(
        "The results are descriptive checks "
        "on the tuning sample only.",
        flush=True,
    )


if __name__ == "__main__":
    main()