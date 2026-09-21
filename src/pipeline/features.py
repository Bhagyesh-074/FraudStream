"""
Feature engineering module for the FraudStream batch pipeline.

This module is deliberately minimal for Phase 2. The most impactful
features for fraud detection — transaction velocity, time-since-last,
merchant-level aggregates — require streaming infrastructure and a feature
store (Phases 3-4) to compute correctly without train/serve inconsistency.

What Phase 2 includes:
  - log_amount: log1p-transformed Amount (handled by cleaning.py)
  - hour_of_day: daily cycle from Time (handled by cleaning.py)
  - V1-V28: PCA features from the dataset provider (used as-is)

What Phase 2 explicitly does NOT include (and why):
  - Transaction velocity (txns/hour per cardholder): requires a streaming
    window or feature store lookup. Engineering a fake version from the
    static dataset would create train/serve inconsistency — the model would
    learn from a feature computed over the full history, but at serving time
    would only have a real-time window. Deferred to Phase 3-4.
  - Time-since-last-transaction: same problem — requires per-cardholder
    state tracking that doesn't exist in a batch CSV.
  - Interaction features between top-effect-size features (V17*V14, etc.):
    XGBoost is a tree-based model that naturally captures feature interactions
    through splits. Adding explicit interaction terms would increase
    dimensionality without a clear benefit and risk overfitting on the small
    fraud class. If we see evidence of missed interactions in Phase 2's
    evaluation, we can revisit.
"""

import pandas as pd
import numpy as np

from src.pipeline.cleaning import (
    prepare_features,
    get_feature_columns,
)


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and target vector from raw data.

    Applies the full cleaning pipeline, then selects the modeling columns.

    Args:
        df: Raw dataset with all original columns including Class.

    Returns:
        Tuple of (X, y):
          X: DataFrame of feature columns (V1-V28, log_amount, hour_of_day)
          y: Series of labels (Class: 0=legit, 1=fraud)
    """
    df_prepared = prepare_features(df)
    feature_cols = get_feature_columns()
    X = df_prepared[feature_cols]
    y = df_prepared["Class"]
    return X, y


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset by time (chronological train/test split).

    Since Time represents seconds since the first transaction, sorting by
    Time and splitting gives a chronological split where the model trains
    on earlier transactions and is tested on later ones — matching how
    fraud detection works in production (you never see the future during
    training).

    This is more realistic than a random stratified split, though it may
    result in slightly different fraud rates between train and test (which
    is itself realistic — fraud patterns shift over time).

    Args:
        df: Raw dataset with a 'Time' column.
        train_fraction: Fraction of data for training (default 0.80).

    Returns:
        Tuple of (train_df, test_df), both containing all original columns.
    """
    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight from the training set's actual ratio.

    NOT hardcoded to 578 (Phase 1's full-dataset ratio). Computed from the
    training split specifically, since the time-based split may yield a
    slightly different ratio.

    Args:
        y_train: Training labels (0/1).

    Returns:
        n_legit / n_fraud from the training set.
    """
    n_fraud = int((y_train == 1).sum())
    n_legit = int((y_train == 0).sum())
    if n_fraud == 0:
        raise ValueError("No fraud samples in training set — cannot compute scale_pos_weight.")
    return n_legit / n_fraud


def split_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Generate a detailed report of the train/test split.

    Includes exact fraud counts (not just rates) and flags if test set
    has too few frauds for reliable evaluation.

    Args:
        train_df: Training split with 'Class' column.
        test_df: Test split with 'Class' column.

    Returns:
        Formatted report string.
    """
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

    # Flag if test set has too few frauds for reliable evaluation
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
