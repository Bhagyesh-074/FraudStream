"""
Full EDA runner for FraudStream Phase 1.

Executes all profiling, cost matrix calibration, and imbalance diagnostics
in sequence. Prints results to stdout and saves key statistics to JSON.

Usage:
    python -m src.eda.run_eda
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.eda.download_data import check_dataset_exists, DEFAULT_CSV_PATH
from src.eda.profiling import (
    load_dataset,
    dataset_summary,
    statistical_profile,
    fraud_vs_legit_comparison,
    amount_outlier_analysis,
)
from src.eda.cost_matrix import CostMatrix
from src.eda.imbalance import (
    compute_imbalance_ratio,
    naive_baseline_accuracy,
    diagnostic_report,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def run_full_eda(data_path: Path = DEFAULT_CSV_PATH) -> dict:
    """Run the complete Phase 1 EDA pipeline.

    Returns:
        Dict of all computed results (for JSON serialization).
    """
    results = {}

    # ── 1. Check dataset exists ──────────────────────────────────────
    print_section("1. DATASET CHECK")
    if not check_dataset_exists(data_path):
        print("ERROR: Dataset not found. Aborting EDA.")
        sys.exit(1)

    # ── 2. Load and summarize ────────────────────────────────────────
    print_section("2. DATASET SUMMARY")
    df = load_dataset(data_path)
    summary = dataset_summary(df)
    results["summary"] = summary

    print(f"Shape:          {summary['n_rows']:,} rows × {summary['n_cols']} columns")
    print(f"Fraud count:    {summary['n_fraud']:,}")
    print(f"Legit count:    {summary['n_legit']:,}")
    print(f"Fraud rate:     {summary['fraud_rate_pct']:.4f}%")
    print(f"Null values:    {sum(summary['null_counts'].values())} total")
    print(f"Columns:        {summary['columns']}")

    # ── 3. Statistical profiling ─────────────────────────────────────
    print_section("3. STATISTICAL PROFILE (all features)")
    profile = statistical_profile(df)
    # Print key features: Amount, Time, and top PCA features
    key_features = ["Amount", "Time", "V1", "V2", "V3", "V14", "V17"]
    available_keys = [f for f in key_features if f in profile.index]
    print(profile.loc[available_keys].to_string())

    # Store all profile data (convert to serializable form)
    results["profile"] = profile.to_dict(orient="index")

    # ── 4. Fraud vs Legit comparison ─────────────────────────────────
    print_section("4. FRAUD vs LEGITIMATE — MEAN COMPARISON")
    comparison = fraud_vs_legit_comparison(df)
    diff_df = comparison["diff"]

    # Sort by absolute effect size to show strongest discriminators
    diff_sorted = diff_df.reindex(
        diff_df["effect_size"].abs().sort_values(ascending=False).index
    )
    print("Top 10 features by effect size (|fraud_mean - legit_mean| / legit_std):")
    print(diff_sorted.head(10).to_string())
    print()
    print("Bottom 5 features (weakest discriminators):")
    print(diff_sorted.tail(5).to_string())

    results["fraud_vs_legit_top10"] = (
        diff_sorted.head(10).to_dict(orient="index")
    )

    # ── 5. Amount outlier analysis ───────────────────────────────────
    print_section("5. AMOUNT OUTLIER ANALYSIS")
    outlier_result = amount_outlier_analysis(df)
    print(outlier_result["reasoning"])
    results["amount_outliers"] = {
        k: v for k, v in outlier_result.items() if k != "reasoning"
    }
    results["amount_outlier_reasoning"] = outlier_result["reasoning"]

    # ── 6. Cost matrix ───────────────────────────────────────────────
    print_section("6. COST MATRIX")
    cost_matrix = CostMatrix()
    cost_matrix.calibrate_from_data(df)
    print(cost_matrix)
    print()
    print(f"Interpretation: Each missed fraud costs ~${cost_matrix.fn_cost:.2f} on average.")
    print(f"Each false alarm costs ~${cost_matrix.fp_cost:.2f} (customer friction).")
    print(f"Cost ratio: {cost_matrix.cost_ratio:.1f}:1 — we should tolerate up to")
    print(f"~{cost_matrix.cost_ratio:.0f} false alarms to avoid missing one fraud.")

    # Example: what if we catch 80% of fraud with 2% false positive rate?
    n_fraud = summary["n_fraud"]
    n_legit = summary["n_legit"]
    tp_example = int(n_fraud * 0.80)
    fn_example = n_fraud - tp_example
    fp_example = int(n_legit * 0.02)
    tn_example = n_legit - fp_example

    example_cost = cost_matrix.compute_expected_cost(
        tp_example, fp_example, fn_example, tn_example
    )
    print()
    print(f"Example scenario (80% recall, 2% FPR):")
    print(f"  TP={tp_example:,}, FP={fp_example:,}, FN={fn_example:,}, TN={tn_example:,}")
    print(f"  FN cost (missed fraud):   ${example_cost['fn_cost']:>12,.2f}")
    print(f"  FP cost (false alarms):   ${example_cost['fp_cost']:>12,.2f}")
    print(f"  Total business cost:      ${example_cost['total_cost']:>12,.2f}")

    results["cost_matrix"] = cost_matrix.summary()
    results["cost_example"] = example_cost

    # ── 7. Imbalance diagnostics ─────────────────────────────────────
    print_section("7. IMBALANCE DIAGNOSTICS")
    print(diagnostic_report(df))

    imb = compute_imbalance_ratio(df)
    baseline = naive_baseline_accuracy(df)
    results["imbalance"] = imb
    results["naive_baseline"] = baseline

    # ── Save results ─────────────────────────────────────────────────
    print_section("RESULTS SAVED")
    output_path = data_path.parent / "eda_results.json"

    # Convert any numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Full EDA results written to: {output_path}")

    return results


if __name__ == "__main__":
    run_full_eda()
