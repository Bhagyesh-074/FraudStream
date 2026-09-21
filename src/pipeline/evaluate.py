"""
Evaluation module for the FraudStream batch pipeline.

Implements cost-based threshold selection using Phase 1's CostMatrix,
AUC-PR as the primary metric, and slice-based evaluation for
Amount bands and hour-of-day segments.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Any

from src.eda.cost_matrix import CostMatrix


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute primary evaluation metrics.

    Args:
        y_true: True labels (0/1).
        y_prob: Predicted probabilities of class 1 (fraud).

    Returns:
        Dict with auc_pr, auc_roc.
    """
    auc_pr = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    return {
        "auc_pr": float(auc_pr),
        "auc_roc": float(auc_roc),
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: CostMatrix,
    n_thresholds: int = 1000,
) -> dict[str, Any]:
    """Find the classification threshold that minimizes total business cost.

    Sweeps thresholds from 0 to 1 in n_thresholds steps, computes the
    confusion matrix and business cost at each, and returns the threshold
    with the lowest total cost.

    This is the core connection between Phase 1's cost matrix and Phase 2's
    model output. Instead of optimizing F1 or accuracy, we optimize
    directly for business cost.

    Args:
        y_true: True labels (0/1).
        y_prob: Predicted probabilities of fraud.
        cost_matrix: Phase 1's CostMatrix instance (calibrated).
        n_thresholds: Number of thresholds to evaluate.

    Returns:
        Dict with:
          optimal_threshold, min_cost, confusion_matrix at optimal,
          cost_at_default_05, all_thresholds (for plotting),
          all_costs (for plotting).
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    costs = []

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
        tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
        result = cost_matrix.compute_expected_cost(tp, fp, fn, tn)
        costs.append(result["total_cost"])

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])

    # Compute confusion matrix at optimal threshold
    y_pred_optimal = (y_prob_arr >= best_threshold).astype(int)
    tp = int(((y_pred_optimal == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_optimal == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_optimal == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_optimal == 0) & (y_true_arr == 0)).sum())

    # Cost at default 0.5 threshold for comparison
    idx_05 = int(n_thresholds * 0.5)
    cost_at_05 = float(costs[idx_05])

    return {
        "optimal_threshold": best_threshold,
        "min_cost": min_cost,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "cost_at_default_05": cost_at_05,
        "all_thresholds": thresholds.tolist(),
        "all_costs": costs.tolist(),
    }


def confusion_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: CostMatrix,
) -> dict[str, Any]:
    """Compute full confusion matrix and cost at a specific threshold.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.
        cost_matrix: Calibrated CostMatrix.

    Returns:
        Dict with tp, fp, fn, tn, precision, recall, f1, and cost breakdown.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cost_result = cost_matrix.compute_expected_cost(tp, fp, fn, tn)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **cost_result,
    }


def slice_evaluation(
    df_test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: CostMatrix,
) -> dict[str, Any]:
    """Evaluate model performance across data slices.

    Slices:
      - Amount bands: [0, 10), [10, 50), [50, 200), [200, 1000), [1000+]
      - Hour of day bands: [0-6), [6-12), [12-18), [18-24)

    Args:
        df_test: Test DataFrame (must contain 'Amount', 'hour_of_day', 'Class').
        y_prob: Predicted probabilities aligned with df_test.
        threshold: Classification threshold.
        cost_matrix: Calibrated CostMatrix.

    Returns:
        Dict with 'by_amount' and 'by_hour' slice results.
    """
    y_true = np.asarray(df_test["Class"])
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    results = {"by_amount": [], "by_hour": []}

    # Amount bands
    amount_bins = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, float("inf"))]
    amount_labels = ["$0-10", "$10-50", "$50-200", "$200-1K", "$1K+"]

    for (lo, hi), label in zip(amount_bins, amount_labels):
        mask = (df_test["Amount"] >= lo) & (df_test["Amount"] < hi)
        idx = mask.values
        if idx.sum() == 0:
            continue

        tp = int(((y_pred[idx] == 1) & (y_true[idx] == 1)).sum())
        fp = int(((y_pred[idx] == 1) & (y_true[idx] == 0)).sum())
        fn = int(((y_pred[idx] == 0) & (y_true[idx] == 1)).sum())
        tn = int(((y_pred[idx] == 0) & (y_true[idx] == 0)).sum())
        n_fraud = int(y_true[idx].sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results["by_amount"].append({
            "band": label,
            "n_total": int(idx.sum()),
            "n_fraud": n_fraud,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
        })

    # Hour-of-day bands
    if "hour_of_day" in df_test.columns:
        hour_bins = [(0, 6), (6, 12), (12, 18), (18, 24)]
        hour_labels = ["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"]

        for (lo, hi), label in zip(hour_bins, hour_labels):
            mask = (df_test["hour_of_day"] >= lo) & (df_test["hour_of_day"] < hi)
            idx = mask.values
            if idx.sum() == 0:
                continue

            tp = int(((y_pred[idx] == 1) & (y_true[idx] == 1)).sum())
            fp = int(((y_pred[idx] == 1) & (y_true[idx] == 0)).sum())
            fn = int(((y_pred[idx] == 0) & (y_true[idx] == 1)).sum())
            tn = int(((y_pred[idx] == 0) & (y_true[idx] == 0)).sum())
            n_fraud = int(y_true[idx].sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            results["by_hour"].append({
                "band": label,
                "n_total": int(idx.sum()),
                "n_fraud": n_fraud,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision,
                "recall": recall,
            })

    return results


def format_evaluation_report(
    metrics: dict,
    threshold_result: dict,
    slices: dict,
    cost_matrix: CostMatrix,
    candidate_thresholds: list[float] | None = None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
) -> str:
    """Format a complete evaluation report with all findings.

    Args:
        metrics: Output of compute_metrics().
        threshold_result: Output of find_cost_optimal_threshold().
        slices: Output of slice_evaluation().
        cost_matrix: Calibrated CostMatrix.
        candidate_thresholds: Optional list of thresholds to show cost at.
        y_true: True labels (needed if candidate_thresholds provided).
        y_prob: Predicted probabilities (needed if candidate_thresholds provided).

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        "EVALUATION REPORT",
        "=" * 70,
        "",
        "PRIMARY METRICS",
        "-" * 40,
        f"  AUC-PR:   {metrics['auc_pr']:.6f}  (random baseline: ~0.0017)",
        f"  AUC-ROC:  {metrics['auc_roc']:.6f}",
        "",
        "COST-OPTIMAL THRESHOLD",
        "-" * 40,
        f"  Optimal threshold:    {threshold_result['optimal_threshold']:.4f}",
        f"  Minimum business cost: ${threshold_result['min_cost']:>12,.2f}",
        f"  Cost at default 0.5:   ${threshold_result['cost_at_default_05']:>12,.2f}",
        f"  Savings vs default:    ${threshold_result['cost_at_default_05'] - threshold_result['min_cost']:>12,.2f}",
        "",
        "CONFUSION MATRIX AT OPTIMAL THRESHOLD",
        "-" * 40,
        f"  True Positives  (caught fraud):  {threshold_result['tp']:>8,}",
        f"  False Positives (false alarms):  {threshold_result['fp']:>8,}",
        f"  False Negatives (missed fraud):  {threshold_result['fn']:>8,}",
        f"  True Negatives  (passed legit):  {threshold_result['tn']:>8,}",
        f"  Precision: {threshold_result['precision']:.4f}",
        f"  Recall:    {threshold_result['recall']:.4f}",
        "",
        f"  FN cost: {threshold_result['fn']} x ${cost_matrix.fn_cost:.2f} = ${threshold_result['fn'] * cost_matrix.fn_cost:>12,.2f}",
        f"  FP cost: {threshold_result['fp']} x ${cost_matrix.fp_cost:.2f} = ${threshold_result['fp'] * cost_matrix.fp_cost:>12,.2f}",
        f"  Total:   ${threshold_result['min_cost']:>12,.2f}",
    ]

    # Cost at candidate thresholds for walkthrough comparison
    if candidate_thresholds and y_true is not None and y_prob is not None:
        lines.extend(["", "COST AT CANDIDATE THRESHOLDS (for comparison)", "-" * 40])
        for t in candidate_thresholds:
            result = confusion_at_threshold(y_true, y_prob, t, cost_matrix)
            lines.append(
                f"  t={t:.2f}: TP={result['tp']:>5}, FP={result['fp']:>6}, "
                f"FN={result['fn']:>4}, TN={result['tn']:>7}  "
                f"Prec={result['precision']:.4f}  Rec={result['recall']:.4f}  "
                f"Cost=${result['total_cost']:>12,.2f}"
            )

    # Slice evaluation
    lines.extend(["", "SLICE EVALUATION: BY AMOUNT BAND", "-" * 40])
    for s in slices.get("by_amount", []):
        lines.append(
            f"  {s['band']:<10} n={s['n_total']:>7,}  fraud={s['n_fraud']:>4}  "
            f"TP={s['tp']:>4}  FP={s['fp']:>6}  FN={s['fn']:>3}  "
            f"Prec={s['precision']:.4f}  Rec={s['recall']:.4f}"
        )

    lines.extend(["", "SLICE EVALUATION: BY HOUR OF DAY", "-" * 40])
    for s in slices.get("by_hour", []):
        lines.append(
            f"  {s['band']:<14} n={s['n_total']:>7,}  fraud={s['n_fraud']:>4}  "
            f"TP={s['tp']:>4}  FP={s['fp']:>6}  FN={s['fn']:>3}  "
            f"Prec={s['precision']:.4f}  Rec={s['recall']:.4f}"
        )

    return "\n".join(lines)


def bootstrap_auc_pr_comparison(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    name_a: str = "XGBoost Baseline",
    name_b: str = "Ensemble",
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare AUC-PR between two models using paired bootstrap resampling.

    Resamples test set predictions with replacement n_bootstraps times.
    On each resample, evaluates AUC-PR for both models and computes their
    paired difference (score_a - score_b).

    Args:
        y_true: True binary labels (0/1).
        y_prob_a: Predicted probabilities from model A.
        y_prob_b: Predicted probabilities from model B.
        name_a: Display name for model A.
        name_b: Display name for model B.
        n_bootstraps: Number of bootstrap iterations (default: 1000).
        random_state: Random seed for exact reproducibility.

    Returns:
        Dict with comprehensive statistics and raw per-iteration scores.
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    prob_a_arr = np.asarray(y_prob_a)
    prob_b_arr = np.asarray(y_prob_b)
    n_samples = len(y_true_arr)

    scores_a = np.empty(n_bootstraps, dtype=float)
    scores_b = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        # Sample indices with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true_arr[idx]

        # Guard against zero-positive resample (probability is negligible on 56k rows, but mathematically sound)
        if y_boot.sum() == 0:
            # Resample until at least one positive is drawn
            while y_boot.sum() == 0:
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                y_boot = y_true_arr[idx]

        scores_a[i] = average_precision_score(y_boot, prob_a_arr[idx])
        scores_b[i] = average_precision_score(y_boot, prob_b_arr[idx])

    diffs = scores_a - scores_b

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "q25": float(np.percentile(arr, 25.0)),
            "q75": float(np.percentile(arr, 75.0)),
        }

    stats_a = _stats(scores_a)
    stats_a["name"] = name_a
    stats_b = _stats(scores_b)
    stats_b["name"] = name_b
    stats_diff = _stats(diffs)

    # Statistical significance: does the 95% CI of diff span across 0?
    statistically_significant = bool(
        (stats_diff["ci_lower"] > 0 and stats_diff["ci_upper"] > 0)
        or (stats_diff["ci_lower"] < 0 and stats_diff["ci_upper"] < 0)
    )

    prob_a_superior = float(np.mean(diffs > 0))
    prob_b_superior = float(np.mean(diffs < 0))
    prob_tie = float(np.mean(diffs == 0))

    return {
        "n_bootstraps": n_bootstraps,
        "model_a": stats_a,
        "model_b": stats_b,
        "diff": stats_diff,
        "prob_a_superior": prob_a_superior,
        "prob_b_superior": prob_b_superior,
        "prob_tie": prob_tie,
        "statistically_significant": statistically_significant,
        "raw_scores_a": scores_a,
        "raw_scores_b": scores_b,
        "raw_diffs": diffs,
    }


def format_bootstrap_report(res: dict[str, Any]) -> str:
    """Format bootstrap model comparison results into a detailed report."""
    ma = res["model_a"]
    mb = res["model_b"]
    d = res["diff"]

    lines = [
        "BOOTSTRAP MODEL COMPARISON REPORT (1,000 PAIRED RESAMPLES)",
        "=" * 70,
        f"Bootstrap iterations:  {res['n_bootstraps']:,}",
        f"Evaluation metric:     AUC-PR (Average Precision)",
        "",
        f"Model A ({ma['name']}):",
        f"  Mean AUC-PR:         {ma['mean']:.6f}  (std: {ma['std']:.6f})",
        f"  Median AUC-PR:       {ma['median']:.6f}",
        f"  95% Bootstrap CI:    [{ma['ci_lower']:.6f}, {ma['ci_upper']:.6f}]",
        f"  IQR (Q25 - Q75):     [{ma['q25']:.6f}, {ma['q75']:.6f}]",
        "",
        f"Model B ({mb['name']}):",
        f"  Mean AUC-PR:         {mb['mean']:.6f}  (std: {mb['std']:.6f})",
        f"  Median AUC-PR:       {mb['median']:.6f}",
        f"  95% Bootstrap CI:    [{mb['ci_lower']:.6f}, {mb['ci_upper']:.6f}]",
        f"  IQR (Q25 - Q75):     [{mb['q25']:.6f}, {mb['q75']:.6f}]",
        "",
        f"Paired Difference ({ma['name']} - {mb['name']}):",
        f"  Mean Difference:     {d['mean']:+.6f}  (std: {d['std']:.6f})",
        f"  Median Difference:   {d['median']:+.6f}",
        f"  95% Bootstrap CI:    [{d['ci_lower']:+.6f}, {d['ci_upper']:+.6f}]",
        f"  IQR (Q25 - Q75):     [{d['q25']:+.6f}, {d['q75']:+.6f}]",
        "",
        f"Hypothesis Testing Assessment:",
        f"  P({ma['name']} > {mb['name']}):  {res['prob_a_superior'] * 100:.2f}%",
        f"  P({mb['name']} > {ma['name']}):  {res['prob_b_superior'] * 100:.2f}%",
        f"  Contains Zero in 95% CI:    {'NO' if res['statistically_significant'] else 'YES'}",
        f"  Statistically Significant:  {'YES (p < 0.05)' if res['statistically_significant'] else 'NO — models are statistically indistinguishable'}",
        "=" * 70,
    ]
    return "\n".join(lines)
