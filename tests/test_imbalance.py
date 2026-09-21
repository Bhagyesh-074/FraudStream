"""Tests for class imbalance diagnostics and baseline accuracy."""

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
    """Create synthetic imbalanced dataset for unit testing."""
    legit = pd.DataFrame({"Class": [0] * 1000})
    fraud = pd.DataFrame({"Class": [1] * 5})
    return pd.concat([legit, fraud], ignore_index=True)


@pytest.fixture
def real_df():
    """Load real dataset, skip if unavailable."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


class TestImbalanceRatioSynthetic:
    """Test imbalance ratio computation on synthetic data."""

    def test_counts(self, sample_df):
        """Verify fraud and legit sample counts."""
        result = compute_imbalance_ratio(sample_df)
        assert result["n_fraud"] == 5 and result["n_legit"] == 1000 and result["n_total"] == 1005

    def test_fraud_rate(self, sample_df):
        """Verify fraud rate fraction matches expectation."""
        result = compute_imbalance_ratio(sample_df)
        assert abs(result["fraud_rate"] - (5 / 1005)) < 1e-10

    def test_imbalance_ratio(self, sample_df):
        """Verify imbalance ratio equals legit / fraud."""
        assert compute_imbalance_ratio(sample_df)["imbalance_ratio"] == 200.0

    def test_scale_pos_weight(self, sample_df):
        """Verify scale_pos_weight matches imbalance ratio."""
        assert compute_imbalance_ratio(sample_df)["scale_pos_weight"] == 200.0


class TestNaiveBaselineSynthetic:
    """Test naive all-legitimate baseline on synthetic data."""

    def test_accuracy(self, sample_df):
        """Verify naive accuracy equals n_legit / n_total."""
        assert abs(naive_baseline_accuracy(sample_df)["accuracy"] - (1000 / 1005)) < 1e-10

    def test_n_wrong_equals_fraud(self, sample_df):
        """Verify misclassifications equal total fraud count."""
        assert naive_baseline_accuracy(sample_df)["n_wrong"] == 5


class TestImbalanceRatioRealData:
    """Verify imbalance statistics against real Kaggle dataset."""

    def test_real_fraud_count(self, real_df):
        """Verify dataset contains 492 fraud transactions."""
        assert compute_imbalance_ratio(real_df)["n_fraud"] == 492

    def test_real_imbalance_ratio(self, real_df):
        """Verify real dataset imbalance ratio is ~578:1."""
        assert 575 < compute_imbalance_ratio(real_df)["imbalance_ratio"] < 580

    def test_real_naive_accuracy_above_99(self, real_df):
        """Verify naive baseline accuracy exceeds 99%."""
        assert naive_baseline_accuracy(real_df)["accuracy_pct"] > 99.0

    def test_real_naive_accuracy_precise(self, real_df):
        """Verify naive baseline accuracy matches ~99.83%."""
        assert 99.80 < naive_baseline_accuracy(real_df)["accuracy_pct"] < 99.85


class TestDiagnosticReport:
    """Test formatted imbalance diagnostic report generation."""

    def test_report_not_empty(self, sample_df):
        """Verify report string is non-empty and descriptive."""
        assert len(diagnostic_report(sample_df)) > 200

    def test_report_contains_key_terms(self, sample_df):
        """Verify report mentions accuracy, fraud, and misleading baseline."""
        report = diagnostic_report(sample_df).lower()
        assert "accuracy" in report and "fraud" in report and "misleading" in report

    def test_report_contains_numbers(self, real_df):
        """Verify real data report contains 492 frauds and 284k transactions."""
        report = diagnostic_report(real_df)
        assert "492" in report and "284" in report
