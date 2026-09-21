"""
Tests for class imbalance diagnostics.

Tests the imbalance ratio calculation, naive baseline accuracy, and
diagnostic report generation.
"""

import pytest
import pandas as pd
import numpy as np

from src.eda.imbalance import (
    compute_imbalance_ratio,
    naive_baseline_accuracy,
    diagnostic_report,
)
from src.eda.download_data import DEFAULT_CSV_PATH


@pytest.fixture
def sample_df():
    """Create a synthetic imbalanced dataset for unit tests."""
    legit = pd.DataFrame({"Class": [0] * 1000})
    fraud = pd.DataFrame({"Class": [1] * 5})
    return pd.concat([legit, fraud], ignore_index=True)


@pytest.fixture
def real_df():
    """Load the real dataset, skip if unavailable."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


class TestImbalanceRatioSynthetic:
    """Test imbalance ratio with known synthetic data."""

    def test_counts(self, sample_df):
        """Should correctly count fraud and legit."""
        result = compute_imbalance_ratio(sample_df)
        assert result["n_fraud"] == 5
        assert result["n_legit"] == 1000
        assert result["n_total"] == 1005

    def test_fraud_rate(self, sample_df):
        """Fraud rate should match expected value."""
        result = compute_imbalance_ratio(sample_df)
        expected_rate = 5 / 1005
        assert abs(result["fraud_rate"] - expected_rate) < 1e-10

    def test_imbalance_ratio(self, sample_df):
        """Imbalance ratio should be 200:1 (1000/5)."""
        result = compute_imbalance_ratio(sample_df)
        assert result["imbalance_ratio"] == 200.0

    def test_scale_pos_weight(self, sample_df):
        """scale_pos_weight should equal legit/fraud."""
        result = compute_imbalance_ratio(sample_df)
        assert result["scale_pos_weight"] == 200.0


class TestNaiveBaselineSynthetic:
    """Test naive baseline accuracy with known data."""

    def test_accuracy(self, sample_df):
        """Accuracy of always-predict-legit should be 1000/1005."""
        result = naive_baseline_accuracy(sample_df)
        expected = 1000 / 1005
        assert abs(result["accuracy"] - expected) < 1e-10

    def test_n_wrong_equals_fraud(self, sample_df):
        """Number of wrong predictions should equal fraud count."""
        result = naive_baseline_accuracy(sample_df)
        assert result["n_wrong"] == 5


class TestImbalanceRatioRealData:
    """Tests against the actual Kaggle dataset."""

    def test_real_fraud_count(self, real_df):
        """Should have exactly 492 fraud transactions."""
        result = compute_imbalance_ratio(real_df)
        assert result["n_fraud"] == 492

    def test_real_imbalance_ratio(self, real_df):
        """Imbalance ratio should be ~577:1."""
        result = compute_imbalance_ratio(real_df)
        # 284315 / 492 ≈ 577.9
        assert 575 < result["imbalance_ratio"] < 580, (
            f"Expected ~578:1, got {result['imbalance_ratio']:.1f}:1"
        )

    def test_real_naive_accuracy_above_99(self, real_df):
        """Naive baseline should be > 99% accurate."""
        result = naive_baseline_accuracy(real_df)
        assert result["accuracy_pct"] > 99.0, (
            f"Expected > 99%, got {result['accuracy_pct']:.4f}%"
        )

    def test_real_naive_accuracy_precise(self, real_df):
        """Naive baseline should be ~99.83% accurate."""
        result = naive_baseline_accuracy(real_df)
        assert 99.80 < result["accuracy_pct"] < 99.85, (
            f"Expected ~99.83%, got {result['accuracy_pct']:.4f}%"
        )


class TestDiagnosticReport:
    """Test the formatted diagnostic report."""

    def test_report_not_empty(self, sample_df):
        """Report should be a substantial string."""
        report = diagnostic_report(sample_df)
        assert len(report) > 200, "Report should be detailed, not a stub"

    def test_report_contains_key_terms(self, sample_df):
        """Report should mention key concepts."""
        report = diagnostic_report(sample_df)
        assert "accuracy" in report.lower()
        assert "fraud" in report.lower()
        assert "misleading" in report.lower()

    def test_report_contains_numbers(self, real_df):
        """Report from real data should contain actual computed values."""
        report = diagnostic_report(real_df)
        assert "492" in report, "Report should mention exact fraud count"
        assert "284" in report, "Report should reference dataset size"
