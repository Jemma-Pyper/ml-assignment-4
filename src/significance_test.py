from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

paired = pd.read_csv(
    RESULTS_DIR / "final_evaluation_paired.csv"
)

comparisons = {
    "Random Forest vs Classification Tree":
        "rf_minus_decision_tree",
    "Random Forest vs k-Nearest Neighbours":
        "rf_minus_knn",
}

results = []

for comparison, column in comparisons.items():
    differences = paired[column].to_numpy()

    n = len(differences)

    mean_difference = differences.mean()
    sd_difference = differences.std(ddof=1)

    # Five-fold CV:
    # validation proportion = 1/5
    # training proportion = 4/5
    # test-to-training ratio = 1/4
    correction = (1 / n) + (1 / 4)

    corrected_standard_error = np.sqrt(
        correction * differences.var(ddof=1)
    )

    t_statistic = (
        mean_difference / corrected_standard_error
    )

    degrees_freedom = n - 1

    p_value = 2 * stats.t.sf(
        abs(t_statistic),
        degrees_freedom,
    )

    results.append({
        "comparison": comparison,
        "mean_macro_f1_difference": mean_difference,
        "sd_difference": sd_difference,
        "t_statistic": t_statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
    })

results_df = pd.DataFrame(results)

output_path = (
    RESULTS_DIR / "final_significance_tests.csv"
)

results_df.to_csv(output_path, index=False)

print(results_df.to_string(index=False))
print(f"\nSaved to: {output_path}")