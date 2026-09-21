"""
Data cleaning module for the FraudStream batch pipeline.

Applies transforms informed by Phase 1's EDA findings:
- Amount: log1p transform (skewness was 16.98 — extremely right-skewed)
- Time: engineer hour_of_day, drop raw Time (monotonic counter, not useful)
- V1-V28: already PCA-transformed/standardized — do NOT re-scale
- Nulls: confirm zero nulls (Phase 1 finding), fail fast if violated

Every transform decision references the Phase 1 analysis that justified it.
"""

import pandas as pd
import numpy as np
from typing import Any


# Feature column definitions
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1-V28
ENGINEERED_FEATURES = ["log_amount", "hour_of_day"]
ALL_FEATURE_COLUMNS = PCA_FEATURES + ENGINEERED_FEATURES


def validate_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm no null values exist in the dataset.

    Phase 1 found zero nulls. This function enforces that assumption
    rather than silently proceeding if the data changes.

    Args:
        df: Raw dataset.

    Returns:
        The same DataFrame (unmodified), after validation.

    Raises:
        ValueError: If any null values are found.
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        offending = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Expected zero nulls (Phase 1 confirmed this), but found "
            f"{total_nulls} nulls in columns: {offending}"
        )
    return df


def validate_no_inf(series: pd.Series, name: str = "column") -> pd.Series:
    """Check that a series contains no infinite values.

    Args:
        series: Data to check.
        name: Column name for error messages.

    Returns:
        The same Series, after validation.

    Raises:
        ValueError: If any Inf/-Inf values are found.
    """
    n_inf = np.isinf(series).sum()
    if n_inf > 0:
        raise ValueError(
            f"Found {n_inf} infinite values in {name} after transform."
        )
    return series


def transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to Amount.

    Why log1p instead of RobustScaler:
      Phase 1 measured Amount skewness at 16.98 — extremely right-skewed.
      log1p(x) = log(1 + x) compresses the long right tail, pulling the
      distribution much closer to normal. RobustScaler would re-center
      using median/IQR but wouldn't change the shape — values would still
      be heavily skewed, and extreme amounts would still dominate gradient
      steps in the autoencoder.

      log1p is also safe for Amount=0 (log1p(0) = 0) and monotonic, so it
      preserves the ordering and the outlier signal that Phase 1 identified
      as correlated with fraud (outlier fraud rate 0.29% vs baseline 0.17%).

    Args:
        df: DataFrame with an 'Amount' column.

    Returns:
        DataFrame with 'log_amount' column added. Original 'Amount' kept
        for reference but excluded from modeling features.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    validate_no_inf(df["log_amount"], "log_amount")
    return df


def engineer_time(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour_of_day from the Time column.

    Time is "seconds since the first transaction in the dataset" over a
    ~48-hour period. Raw Time is a monotonic counter encoding position in
    the dataset — it would leak temporal ordering and not generalize to
    new data.

    Instead, we extract hour_of_day = (Time % 86400) / 3600, which
    captures the cyclic daily pattern. Fraud patterns are known to cluster
    at certain hours (e.g., late night when cardholders are asleep and
    less likely to notice unauthorized charges).

    Args:
        df: DataFrame with a 'Time' column (seconds since first transaction).

    Returns:
        DataFrame with 'hour_of_day' column added (0.0 to 23.99...).
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) / 3600.0
    return df


def confirm_pca_features_prescaled(df: pd.DataFrame) -> dict[str, Any]:
    """Confirm V1-V28 are already standardized and should NOT be re-scaled.

    Phase 1's statistical profile showed V1-V28 have means ≈ 0 and stds
    in the range ~0.5-2.0, consistent with PCA-transformed features that
    are already standardized by the dataset provider.

    Returns:
        Dict with per-feature mean and std for verification logging.
    """
    stats = {}
    for col in PCA_FEATURES:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    return stats


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: validate, transform, engineer.

    Steps:
    1. Validate no nulls
    2. Transform Amount -> log_amount
    3. Engineer Time -> hour_of_day
    4. Confirm V1-V28 are pre-scaled (log, don't re-scale)

    Args:
        df: Raw dataset with all original columns.

    Returns:
        DataFrame with all original columns PLUS engineered columns.
        Callers use get_feature_columns() to select modeling features.
    """
    df = validate_no_nulls(df)
    df = transform_amount(df)
    df = engineer_time(df)
    # V1-V28 are kept as-is (already PCA-scaled)
    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature columns used for modeling.

    Excludes: Time (raw monotonic counter), Amount (replaced by log_amount),
    Class (the target label).

    Includes: V1-V28 (PCA features), log_amount, hour_of_day.
    """
    return ALL_FEATURE_COLUMNS.copy()
