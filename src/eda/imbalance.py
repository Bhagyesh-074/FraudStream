"""Class imbalance diagnostics for fraud detection."""

import pandas as pd
import numpy as np
from typing import Any


def compute_imbalance_ratio(df: pd.DataFrame) -> dict[str, Any]:
    """Compute fraud vs legitimate class counts and imbalance ratio."""
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())
    n_total = len(df)
    fraud_rate = n_fraud / n_total

    return {
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "n_total": n_total,
        "fraud_rate": fraud_rate,
        "fraud_rate_pct": fraud_rate * 100,
        "imbalance_ratio": n_legit / n_fraud if n_fraud > 0 else float("inf"),
        "scale_pos_weight": n_legit / n_fraud if n_fraud > 0 else 1.0,
    }


def naive_baseline_accuracy(df: pd.DataFrame) -> dict[str, Any]:
    """Compute accuracy of a majority-class (always legitimate) baseline."""
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())
    n_total = len(df)
    accuracy = n_legit / n_total

    return {
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "n_correct": n_legit,
        "n_wrong": n_fraud,
    }


def diagnostic_report(df: pd.DataFrame) -> str:
    """Generate diagnostic report comparing imbalance against naive baseline."""
    imb = compute_imbalance_ratio(df)
    baseline = naive_baseline_accuracy(df)

    lines = [
        "CLASS IMBALANCE DIAGNOSTIC REPORT",
        "=" * 60,
        "",
        f"Total transactions:       {imb['n_total']:>10,}",
        f"Legitimate (Class=0):     {imb['n_legit']:>10,}",
        f"Fraudulent (Class=1):     {imb['n_fraud']:>10,}",
        f"Fraud rate:               {imb['fraud_rate_pct']:>10.4f}%",
        f"Imbalance ratio:          {imb['imbalance_ratio']:>10.1f}:1  (legit:fraud)",
        "",
        "NAIVE BASELINE (always predict 'not fraud'):",
        f"  Accuracy:               {baseline['accuracy_pct']:.4f}%",
        f"  Correct predictions:    {baseline['n_correct']:,}",
        f"  Missed frauds:          {baseline['n_wrong']:,}  (ALL of them)",
        "",
        "WHY ACCURACY IS MISLEADING:",
        f"  A model that NEVER predicts fraud achieves {baseline['accuracy_pct']:.2f}% accuracy.",
        f"  This sounds impressive but misses every single fraud —",
        f"  {imb['n_fraud']} out of {imb['n_fraud']} fraudulent transactions go undetected.",
        f"  With ~{imb['fraud_rate_pct']:.2f}% fraud, the baseline accuracy is ~{baseline['accuracy_pct']:.2f}%.",
        f"  Any model must beat this trivial baseline on MEANINGFUL metrics",
        f"  (precision, recall, AUC-PR, business cost), not on accuracy.",
        "",
        "IMPLICATIONS FOR PHASE 2 (noted, not implemented yet):",
        f"  - XGBoost scale_pos_weight should be ~{imb['scale_pos_weight']:.0f}",
        f"    (weight fraud class {imb['scale_pos_weight']:.0f}x to compensate for rarity)",
        f"  - Alternatively, SMOTE or random undersampling of the majority class",
        f"  - Evaluation must use precision-recall curves, not ROC alone",
        f"  - Cost matrix will drive optimal threshold selection",
    ]

    return "\n".join(lines)
