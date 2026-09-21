"""Data cleaning and feature transformation module for FraudStream."""

import pandas as pd
import numpy as np
from typing import Any

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_FEATURES = ["log_amount", "hour_of_day"]
ALL_FEATURE_COLUMNS = PCA_FEATURES + ENGINEERED_FEATURES


def validate_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that dataframe contains zero null values."""
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        offending = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Found {total_nulls} nulls in columns: {offending}")
    return df


def validate_no_inf(series: pd.Series, name: str = "column") -> pd.Series:
    """Check that series contains no infinite values."""
    n_inf = np.isinf(series).sum()
    if n_inf > 0:
        raise ValueError(f"Found {n_inf} infinite values in {name} after transform.")
    return series


def transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to Amount column."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    validate_no_inf(df["log_amount"], "log_amount")
    return df


def engineer_time(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclic hour_of_day (0 to 24) from Time column."""
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) / 3600.0
    return df


def confirm_pca_features_prescaled(df: pd.DataFrame) -> dict[str, Any]:
    """Verify mean and standard deviation for pre-scaled PCA features."""
    stats = {}
    for col in PCA_FEATURES:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    return stats


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run data validation, Amount log1p, and hour_of_day engineering."""
    df = validate_no_nulls(df)
    df = transform_amount(df)
    df = engineer_time(df)
    return df


def get_feature_columns() -> list[str]:
    """Return list of 30 feature column names used for model training."""
    return ALL_FEATURE_COLUMNS.copy()
