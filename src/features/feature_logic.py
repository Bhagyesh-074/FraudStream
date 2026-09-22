"""Centralized feature transformation module ensuring train/serve consistency."""

from typing import Any
import numpy as np
import pandas as pd

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_FEATURES = ["log_amount", "hour_of_day"]
FEATURE_COLUMNS = PCA_FEATURES + ENGINEERED_FEATURES
REQUIRED_RAW_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount"]


def compute_features(data: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Compute the 30 features from raw record (dict/Series) or batch DataFrame."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data.to_dict()])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(data)}")

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for feature computation: {missing}")

    if df[REQUIRED_RAW_COLUMNS].isna().any().any():
        nulls = df[REQUIRED_RAW_COLUMNS].isnull().sum()
        raise ValueError(f"Found null values in raw columns: {nulls[nulls > 0].to_dict()}")

    df["log_amount"] = np.log1p(df["Amount"])
    if np.isinf(df["log_amount"]).any():
        raise ValueError("Infinite values produced during log1p Amount transform.")

    df["hour_of_day"] = (df["Time"] % 86400) / 3600.0
    return df[FEATURE_COLUMNS]
