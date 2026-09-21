"""Feature matrix construction and chronological train/test split."""

import pandas as pd
import numpy as np

from src.pipeline.cleaning import (
    prepare_features,
    get_feature_columns,
)


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X (30 columns) and target vector y from dataset."""
    df_prepared = prepare_features(df)
    feature_cols = get_feature_columns()
    X = df_prepared[feature_cols]
    y = df_prepared["Class"]
    return X, y


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform chronological train/test split sorted by Time column."""
    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute scale_pos_weight (legit/fraud ratio) from training split."""
    n_fraud = int((y_train == 1).sum())
    n_legit = int((y_train == 0).sum())
    if n_fraud == 0:
        raise ValueError("No fraud samples in training set — cannot compute scale_pos_weight.")
    return n_legit / n_fraud


def split_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Generate train/test split report with class counts and size warnings."""
    train_fraud = int((train_df["Class"] == 1).sum())
    train_legit = int((train_df["Class"] == 0).sum())
    test_fraud = int((test_df["Class"] == 1).sum())
    test_legit = int((test_df["Class"] == 0).sum())

    train_rate = train_fraud / len(train_df) * 100
    test_rate = test_fraud / len(test_df) * 100

    lines = [
        "TIME-BASED TRAIN/TEST SPLIT REPORT",
        "=" * 60,
        "",
        f"Train set:  {len(train_df):>10,} rows",
        f"  Legit:    {train_legit:>10,}",
        f"  Fraud:    {train_fraud:>10,}  ({train_rate:.4f}%)",
        f"  Ratio:    {train_legit/train_fraud:.1f}:1  (legit:fraud)",
        "",
        f"Test set:   {len(test_df):>10,} rows",
        f"  Legit:    {test_legit:>10,}",
        f"  Fraud:    {test_fraud:>10,}  ({test_rate:.4f}%)",
        f"  Ratio:    {test_legit/test_fraud:.1f}:1  (legit:fraud)" if test_fraud > 0 else "  Ratio:    N/A (no fraud)",
        "",
        f"Time range — Train: [{train_df['Time'].min():.0f}, {train_df['Time'].max():.0f}]",
        f"Time range — Test:  [{test_df['Time'].min():.0f}, {test_df['Time'].max():.0f}]",
    ]

    FRAUD_COUNT_THRESHOLD = 80
    if test_fraud < FRAUD_COUNT_THRESHOLD:
        lines.extend([
            "",
            "*** WARNING: LOW FRAUD COUNT IN TEST SET ***",
            f"  Test set has only {test_fraud} fraud transactions.",
            f"  With fewer than {FRAUD_COUNT_THRESHOLD} fraud samples, threshold-selection",
            "  and evaluation metrics may be unreliable due to high variance.",
            "  Precision/recall estimates at specific thresholds will have wide",
            "  confidence intervals. Treat per-threshold numbers as approximate.",
            "  Consider: cross-validation on the training set for more stable",
            "  threshold tuning, using test set only for final hold-out reporting.",
        ])
    else:
        lines.extend([
            "",
            f"  Test fraud count ({test_fraud}) is above threshold ({FRAUD_COUNT_THRESHOLD}).",
            "  Evaluation metrics should be reasonably stable.",
        ])

    return "\n".join(lines)
