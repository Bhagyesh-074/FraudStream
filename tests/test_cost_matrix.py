"""
Tests for the cost matrix module.

Tests that cost values are set, non-trivial, correctly calibrated from
data, and that the expected cost computation works as intended.
"""

import pytest
import pandas as pd
import numpy as np

from src.eda.cost_matrix import CostMatrix
from src.eda.download_data import DEFAULT_CSV_PATH


@pytest.fixture
def sample_df():
    """Create a small synthetic dataset for unit tests.

    This avoids requiring the full Kaggle dataset for cost matrix tests.
    Mimics the structure: fraud amounts average $150, legit average $88.
    """
    np.random.seed(42)
    n_legit = 1000
    n_fraud = 5

    legit = pd.DataFrame({
        "Amount": np.random.exponential(88, n_legit),
        "Class": 0,
    })
    fraud = pd.DataFrame({
        "Amount": np.random.exponential(150, n_fraud),
        "Class": 1,
    })
    return pd.concat([legit, fraud], ignore_index=True)


@pytest.fixture
def real_df():
    """Load the real dataset, skip if unavailable."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


class TestCostMatrixDefaults:
    """Test default cost matrix initialization."""

    def test_fp_cost_is_positive(self):
        """FP cost should be set to a positive value by default."""
        cm = CostMatrix()
        assert cm.fp_cost > 0, "FP cost should be positive"

    def test_fp_cost_is_nontrivial(self):
        """FP cost should be a reasonable value, not $0.01 or $10000."""
        cm = CostMatrix()
        assert 1.0 <= cm.fp_cost <= 50.0, (
            f"FP cost ${cm.fp_cost} seems unreasonable for customer friction"
        )

    def test_tp_cost_is_zero(self):
        """TP cost should be $0 (catching fraud has no business cost)."""
        cm = CostMatrix()
        assert cm.tp_cost == 0.0

    def test_tn_cost_is_zero(self):
        """TN cost should be $0 (correctly passing legit has no cost)."""
        cm = CostMatrix()
        assert cm.tn_cost == 0.0


class TestCostMatrixCalibration:
    """Test cost matrix calibration from data."""

    def test_calibrate_sets_fn_cost(self, sample_df):
        """After calibration, FN cost should be set to avg fraud amount."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        assert cm.fn_cost is not None, "FN cost should be set after calibration"
        assert cm.fn_cost > 0, "FN cost should be positive"

    def test_fn_cost_matches_avg_fraud(self, sample_df):
        """FN cost should equal the average fraud transaction amount."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        expected = sample_df.loc[sample_df["Class"] == 1, "Amount"].mean()
        assert abs(cm.fn_cost - expected) < 0.01, (
            f"FN cost ${cm.fn_cost:.2f} != avg fraud ${expected:.2f}"
        )

    def test_fn_cost_greater_than_fp_cost(self, sample_df):
        """Missed fraud should cost more than a false alarm."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        assert cm.fn_cost > cm.fp_cost, (
            f"FN cost (${cm.fn_cost:.2f}) should exceed FP cost (${cm.fp_cost:.2f})"
        )

    def test_cost_ratio_positive(self, sample_df):
        """Cost ratio should be positive and > 1."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        assert cm.cost_ratio > 1.0, (
            f"Cost ratio {cm.cost_ratio:.1f} should be > 1"
        )


class TestComputeExpectedCost:
    """Test the expected cost computation."""

    def test_all_correct_costs_zero(self, sample_df):
        """Perfect predictions should have $0 total cost."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=5, fp=0, fn=0, tn=1000)
        assert result["total_cost"] == 0.0

    def test_all_missed_costs_sum(self, sample_df):
        """Missing all frauds should cost fn_cost * n_fraud."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=0, fp=0, fn=5, tn=1000)
        expected = 5 * cm.fn_cost
        assert abs(result["total_cost"] - expected) < 0.01

    def test_false_alarms_cost(self, sample_df):
        """False alarms should cost fp_cost * n_fp."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=5, fp=100, fn=0, tn=900)
        expected = 100 * cm.fp_cost
        assert abs(result["total_cost"] - expected) < 0.01

    def test_mixed_scenario(self, sample_df):
        """Mixed confusion matrix should compute correctly."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=3, fp=50, fn=2, tn=950)
        expected = (2 * cm.fn_cost) + (50 * cm.fp_cost)
        assert abs(result["total_cost"] - expected) < 0.01

    def test_result_keys(self, sample_df):
        """Result should contain all expected keys."""
        cm = CostMatrix()
        cm.calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=1, fp=1, fn=1, tn=1)
        expected_keys = [
            "tp_cost", "fp_cost", "fn_cost", "tn_cost", "total_cost",
            "tp_count", "fp_count", "fn_count", "tn_count",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestCostMatrixWithRealData:
    """Tests against the actual Kaggle dataset."""

    def test_real_fn_cost_reasonable(self, real_df):
        """FN cost from real data should be in a reasonable range."""
        cm = CostMatrix()
        cm.calibrate_from_data(real_df)
        # Avg fraud amount in this dataset is historically ~$122
        assert 50 < cm.fn_cost < 300, (
            f"FN cost ${cm.fn_cost:.2f} outside expected range for this dataset"
        )

    def test_real_cost_ratio(self, real_df):
        """Cost ratio from real data should be meaningful (> 5:1)."""
        cm = CostMatrix()
        cm.calibrate_from_data(real_df)
        assert cm.cost_ratio > 5, (
            f"Cost ratio {cm.cost_ratio:.1f}:1 seems too low"
        )
