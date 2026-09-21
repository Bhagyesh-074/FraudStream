"""
Tests for dataset loading and statistical profiling.

These tests require the dataset to be present at data/creditcard.csv.
If the file is missing, tests are skipped with a descriptive message.
"""

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


# Skip all tests if dataset is not available
pytestmark = pytest.mark.skipif(
    not DEFAULT_CSV_PATH.is_file(),
    reason=f"Dataset not found at {DEFAULT_CSV_PATH}. Download from Kaggle first.",
)


@pytest.fixture(scope="module")
def df():
    """Load dataset once for all tests in this module."""
    return load_dataset(DEFAULT_CSV_PATH)


class TestDatasetLoading:
    """Tests for dataset shape and basic integrity."""

    def test_shape_rows(self, df):
        """Dataset should have ~284,807 rows."""
        assert df.shape[0] == 284807, (
            f"Expected 284,807 rows, got {df.shape[0]:,}"
        )

    def test_shape_cols(self, df):
        """Dataset should have 31 columns (Time, V1-V28, Amount, Class)."""
        assert df.shape[1] == 31, (
            f"Expected 31 columns, got {df.shape[1]}"
        )

    def test_expected_columns(self, df):
        """Dataset should contain Time, Amount, Class, and V1-V28."""
        expected = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_values(self, df):
        """This dataset is known to have no nulls."""
        total_nulls = df.isnull().sum().sum()
        assert total_nulls == 0, f"Found {total_nulls} null values"

    def test_class_values_binary(self, df):
        """Class column should contain only 0 and 1."""
        unique_classes = set(df["Class"].unique())
        assert unique_classes == {0, 1}, (
            f"Expected classes {{0, 1}}, got {unique_classes}"
        )


class TestFraudCount:
    """Tests for fraud/legitimate transaction counts."""

    def test_fraud_count(self, df):
        """Should have exactly 492 fraud transactions."""
        n_fraud = (df["Class"] == 1).sum()
        assert n_fraud == 492, f"Expected 492 frauds, got {n_fraud}"

    def test_legit_count(self, df):
        """Should have exactly 284,315 legitimate transactions."""
        n_legit = (df["Class"] == 0).sum()
        assert n_legit == 284315, f"Expected 284,315 legit, got {n_legit}"


class TestDatasetSummary:
    """Tests for the dataset_summary function."""

    def test_summary_keys(self, df):
        """Summary should contain all expected keys."""
        summary = dataset_summary(df)
        expected_keys = [
            "n_rows", "n_cols", "n_fraud", "n_legit",
            "fraud_rate", "fraud_rate_pct", "dtypes",
            "null_counts", "columns",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_fraud_rate_range(self, df):
        """Fraud rate should be between 0.1% and 0.2%."""
        summary = dataset_summary(df)
        assert 0.1 < summary["fraud_rate_pct"] < 0.2, (
            f"Fraud rate {summary['fraud_rate_pct']:.4f}% outside expected range"
        )


class TestStatisticalProfile:
    """Tests for statistical profiling output."""

    def test_profile_has_all_features(self, df):
        """Profile should include all 30 features (Time, V1-V28, Amount)."""
        profile = statistical_profile(df)
        assert len(profile) == 30, f"Expected 30 features, got {len(profile)}"

    def test_profile_has_all_stats(self, df):
        """Each feature should have all statistical columns."""
        profile = statistical_profile(df)
        expected_stats = [
            "mean", "std", "median", "min", "max",
            "skewness", "kurtosis", "p1", "p5", "p25", "p75", "p95", "p99",
        ]
        for stat in expected_stats:
            assert stat in profile.columns, f"Missing stat column: {stat}"

    def test_amount_stats_reasonable(self, df):
        """Amount mean should be positive and finite."""
        profile = statistical_profile(df)
        amount_mean = profile.loc["Amount", "mean"]
        assert amount_mean > 0, "Amount mean should be positive"
        assert np.isfinite(amount_mean), "Amount mean should be finite"


class TestFraudVsLegitComparison:
    """Tests for fraud vs legitimate comparison."""

    def test_comparison_keys(self, df):
        """Should return fraud, legit, and diff profiles."""
        comp = fraud_vs_legit_comparison(df)
        assert "fraud" in comp
        assert "legit" in comp
        assert "diff" in comp

    def test_diff_has_effect_size(self, df):
        """Diff DataFrame should include effect size column."""
        comp = fraud_vs_legit_comparison(df)
        assert "effect_size" in comp["diff"].columns


class TestAmountOutlierAnalysis:
    """Tests for amount outlier analysis."""

    def test_outlier_analysis_keys(self, df):
        """Should return all expected keys."""
        result = amount_outlier_analysis(df)
        expected_keys = [
            "q1", "q3", "iqr", "upper_fence", "p99",
            "fraud_rate_outlier_iqr", "fraud_rate_normal",
            "reasoning",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_reasoning_not_empty(self, df):
        """Reasoning string should be substantial, not just a placeholder."""
        result = amount_outlier_analysis(df)
        assert len(result["reasoning"]) > 100, (
            "Reasoning should be a detailed explanation, not a stub"
        )
