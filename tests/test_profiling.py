"""Tests for dataset loading and statistical profiling."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.eda.profiling import (
    load_dataset,
    dataset_summary,
    statistical_profile,
    fraud_vs_legit_comparison,
    amount_outlier_analysis,
)
from src.eda.download_data import DEFAULT_CSV_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_CSV_PATH.is_file(),
    reason=f"Dataset not found at {DEFAULT_CSV_PATH}.",
)


@pytest.fixture(scope="module")
def df():
    """Load dataset once for module test suite."""
    return load_dataset(DEFAULT_CSV_PATH)


class TestDatasetLoading:
    """Verify dataset shape and basic integrity."""

    def test_shape_rows(self, df):
        """Verify row count matches 284,807."""
        assert df.shape[0] == 284807

    def test_shape_cols(self, df):
        """Verify column count matches 31."""
        assert df.shape[1] == 31

    def test_expected_columns(self, df):
        """Verify presence of Time, Amount, Class, and V1-V28."""
        for col in ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]:
            assert col in df.columns

    def test_no_null_values(self, df):
        """Verify dataset contains zero null values."""
        assert df.isnull().sum().sum() == 0

    def test_class_values_binary(self, df):
        """Verify Class column only contains 0 and 1."""
        assert set(df["Class"].unique()) == {0, 1}


class TestFraudCount:
    """Verify fraud and legitimate transaction class counts."""

    def test_fraud_count(self, df):
        """Verify exactly 492 fraud samples."""
        assert (df["Class"] == 1).sum() == 492

    def test_legit_count(self, df):
        """Verify exactly 284,315 legitimate samples."""
        assert (df["Class"] == 0).sum() == 284315


class TestDatasetSummary:
    """Verify dataset summary output."""

    def test_summary_keys(self, df):
        """Verify summary output dictionary keys."""
        summary = dataset_summary(df)
        for key in ["n_rows", "n_cols", "n_fraud", "n_legit", "fraud_rate", "fraud_rate_pct", "dtypes", "null_counts", "columns"]:
            assert key in summary

    def test_fraud_rate_range(self, df):
        """Verify fraud rate is approximately 0.17%."""
        assert 0.1 < dataset_summary(df)["fraud_rate_pct"] < 0.2


class TestStatisticalProfile:
    """Verify statistical profile generation across all numeric features."""

    def test_profile_has_all_features(self, df):
        """Verify profile covers all 30 non-target features."""
        assert len(statistical_profile(df)) == 30

    def test_profile_has_all_stats(self, df):
        """Verify profile includes standard summary metrics and percentiles."""
        profile = statistical_profile(df)
        for stat in ["mean", "std", "median", "min", "max", "skewness", "kurtosis", "p1", "p5", "p25", "p75", "p95", "p99"]:
            assert stat in profile.columns

    def test_amount_stats_reasonable(self, df):
        """Verify Amount mean is positive and finite."""
        mean = statistical_profile(df).loc["Amount", "mean"]
        assert mean > 0 and np.isfinite(mean)


class TestFraudVsLegitComparison:
    """Verify group comparison and effect size calculation."""

    def test_comparison_keys(self, df):
        """Verify output dictionary contains fraud, legit, and diff frames."""
        comp = fraud_vs_legit_comparison(df)
        assert "fraud" in comp and "legit" in comp and "diff" in comp

    def test_diff_has_effect_size(self, df):
        """Verify diff dataframe contains effect_size column."""
        assert "effect_size" in fraud_vs_legit_comparison(df)["diff"].columns


class TestAmountOutlierAnalysis:
    """Verify amount outlier detection and reasoning."""

    def test_outlier_analysis_keys(self, df):
        """Verify outlier analysis dictionary keys."""
        res = amount_outlier_analysis(df)
        for key in ["q1", "q3", "iqr", "upper_fence", "p99", "fraud_rate_outlier_iqr", "fraud_rate_normal", "reasoning"]:
            assert key in res

    def test_reasoning_not_empty(self, df):
        """Verify reasoning string is non-empty."""
        assert len(amount_outlier_analysis(df)["reasoning"]) > 50
