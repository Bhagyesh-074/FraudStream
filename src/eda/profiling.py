"""
Statistical profiling for the Credit Card Fraud dataset.

Provides functions for loading data, computing distribution statistics,
and comparing fraud vs legitimate transaction profiles. All analysis
functions return plain data structures (dicts, DataFrames) for testability
and reuse by run_eda.py and downstream phases.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Any

from src.eda.download_data import DEFAULT_CSV_PATH


def load_dataset(path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV.

    Args:
        path: Path to creditcard.csv.

    Returns:
        DataFrame with all columns (Time, V1-V28, Amount, Class).

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"See README.md for download instructions."
        )
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute basic dataset summary statistics.

    Returns:
        Dict with keys: n_rows, n_cols, n_fraud, n_legit, fraud_rate,
        dtypes (dict), null_counts (dict), columns (list).
    """
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())
    n_rows, n_cols = df.shape

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate": n_fraud / n_rows,
        "fraud_rate_pct": (n_fraud / n_rows) * 100,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "columns": list(df.columns),
    }


def statistical_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution statistics for all numeric features.

    For each feature, computes: mean, std, median, min, max, skewness,
    kurtosis, and select percentiles.

    Returns:
        DataFrame indexed by feature name with stat columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude Class from profiling (it's a label, not a feature)
    feature_cols = [c for c in numeric_cols if c != "Class"]

    records = []
    for col in feature_cols:
        series = df[col]
        records.append({
            "feature": col,
            "mean": series.mean(),
            "std": series.std(),
            "median": series.median(),
            "min": series.min(),
            "max": series.max(),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "p1": series.quantile(0.01),
            "p5": series.quantile(0.05),
            "p25": series.quantile(0.25),
            "p75": series.quantile(0.75),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
        })

    return pd.DataFrame(records).set_index("feature")


def fraud_vs_legit_comparison(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare statistical profiles of fraud vs legitimate transactions.

    Returns:
        Dict with keys 'fraud' and 'legit', each a profile DataFrame,
        plus 'diff' with the mean difference (fraud - legit) for each feature.
    """
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    fraud_profile = statistical_profile(fraud_df)
    legit_profile = statistical_profile(legit_df)

    # Mean difference highlights which features diverge most between classes
    mean_diff = fraud_profile["mean"] - legit_profile["mean"]
    # Normalized by legit std to show effect size
    legit_std = legit_profile["std"].replace(0, np.nan)
    effect_size = mean_diff / legit_std

    diff_df = pd.DataFrame({
        "fraud_mean": fraud_profile["mean"],
        "legit_mean": legit_profile["mean"],
        "mean_diff": mean_diff,
        "effect_size": effect_size,
        "fraud_std": fraud_profile["std"],
        "legit_std": legit_profile["std"],
    })

    return {
        "fraud": fraud_profile,
        "legit": legit_profile,
        "diff": diff_df,
    }


def amount_outlier_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze Amount outliers with explicit reasoning about signal vs noise.

    Computes IQR-based and percentile-based outlier thresholds, then
    compares fraud rates among outliers vs non-outliers to determine
    whether extreme amounts are plausible fraud signal or data error.

    Returns:
        Dict with outlier thresholds, counts, fraud rates by group,
        and a reasoning string.
    """
    amount = df["Amount"]

    # IQR method
    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Percentile method
    p99 = amount.quantile(0.99)
    p999 = amount.quantile(0.999)

    # Identify outliers (IQR upper fence — lower fence is typically 0 for
    # Amount since Q1 is often small)
    is_outlier_iqr = amount > upper_fence
    is_extreme = amount > p99

    # Fraud rates among outlier groups
    n_outlier_iqr = int(is_outlier_iqr.sum())
    n_extreme = int(is_extreme.sum())

    fraud_rate_outlier_iqr = (
        df.loc[is_outlier_iqr, "Class"].mean() if n_outlier_iqr > 0 else 0.0
    )
    fraud_rate_extreme = (
        df.loc[is_extreme, "Class"].mean() if n_extreme > 0 else 0.0
    )
    fraud_rate_normal = (
        df.loc[~is_outlier_iqr, "Class"].mean() if (~is_outlier_iqr).sum() > 0 else 0.0
    )
    overall_fraud_rate = df["Class"].mean()

    # Fraud-specific amount stats
    fraud_amounts = df.loc[df["Class"] == 1, "Amount"]
    legit_amounts = df.loc[df["Class"] == 0, "Amount"]

    # Build reasoning
    reasoning_parts = [
        "AMOUNT OUTLIER ANALYSIS — Signal vs Data Error Assessment",
        "=" * 60,
        "",
        f"IQR fences: lower={lower_fence:.2f}, upper={upper_fence:.2f}",
        f"IQR-based outliers (Amount > {upper_fence:.2f}): {n_outlier_iqr:,} transactions",
        f"Extreme values (Amount > p99={p99:.2f}): {n_extreme:,} transactions",
        f"Top value (p99.9): {p999:.2f}, absolute max: {amount.max():.2f}",
        "",
        f"Fraud rate among IQR outliers: {fraud_rate_outlier_iqr:.4%}",
        f"Fraud rate among non-outliers:  {fraud_rate_normal:.4%}",
        f"Fraud rate among p99+ extremes: {fraud_rate_extreme:.4%}",
        f"Overall fraud rate:             {overall_fraud_rate:.4%}",
        "",
        f"Fraud Amount — mean: {fraud_amounts.mean():.2f}, median: {fraud_amounts.median():.2f}, "
        f"max: {fraud_amounts.max():.2f}, std: {fraud_amounts.std():.2f}",
        f"Legit Amount — mean: {legit_amounts.mean():.2f}, median: {legit_amounts.median():.2f}, "
        f"max: {legit_amounts.max():.2f}, std: {legit_amounts.std():.2f}",
        "",
        "REASONING:",
        "  - Credit card transactions naturally span a wide range ($0.01 to",
        "    thousands), so high-Amount transactions are not inherently erroneous.",
        "  - If outlier fraud rate is HIGHER than baseline, extreme amounts are",
        "    correlated with fraud and are useful signal — do not remove.",
        "  - If outlier fraud rate is SIMILAR or LOWER, extreme amounts are just",
        "    large legitimate purchases — still not data errors, just noise.",
        "  - Either way, Amount outliers in credit card data are plausible real",
        "    transactions, not data entry errors. The key question for Phase 2",
        "    is whether to scale/clip Amount for model stability, not whether",
        "    to drop outlier rows.",
    ]

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "p99": float(p99),
        "p999": float(p999),
        "max_amount": float(amount.max()),
        "n_outlier_iqr": n_outlier_iqr,
        "n_extreme_p99": n_extreme,
        "fraud_rate_outlier_iqr": float(fraud_rate_outlier_iqr),
        "fraud_rate_normal": float(fraud_rate_normal),
        "fraud_rate_extreme_p99": float(fraud_rate_extreme),
        "overall_fraud_rate": float(overall_fraud_rate),
        "fraud_amount_mean": float(fraud_amounts.mean()),
        "fraud_amount_median": float(fraud_amounts.median()),
        "fraud_amount_max": float(fraud_amounts.max()),
        "fraud_amount_std": float(fraud_amounts.std()),
        "legit_amount_mean": float(legit_amounts.mean()),
        "legit_amount_median": float(legit_amounts.median()),
        "legit_amount_max": float(legit_amounts.max()),
        "legit_amount_std": float(legit_amounts.std()),
        "reasoning": "\n".join(reasoning_parts),
    }
