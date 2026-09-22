"""Data cleaning and feature transformation module for FraudStream."""

from typing import Any
import numpy as np
import pandas as pd
from src.features.feature_logic import (
    PCA_FEATURES,
    ENGINEERED_FEATURES,
    FEATURE_COLUMNS as ALL_FEATURE_COLUMNS,
    compute_features,
)


def validate_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that dataframe contains zero null values."""
    if df.isna().any().any():
        nulls = df.isnull().sum()
        raise ValueError(f"Found {nulls.sum()} nulls in columns: {nulls[nulls > 0].to_dict()}")
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
    return {col: {"mean": float(df[col].mean()), "std": float(df[col].std())} for col in PCA_FEATURES if col in df.columns}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe by applying shared feature computation logic."""
    feats = compute_features(df)
    df_out = df.copy()
    df_out["log_amount"] = feats["log_amount"]
    df_out["hour_of_day"] = feats["hour_of_day"]
    return df_out


def get_feature_columns() -> list[str]:
    """Return list of 30 feature column names used for model training."""
    return ALL_FEATURE_COLUMNS.copy()


