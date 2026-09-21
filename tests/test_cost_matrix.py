"""Tests for cost matrix and expected cost computation."""

import pytest
import pandas as pd
import numpy as np

from src.eda.cost_matrix import CostMatrix
from src.eda.download_data import DEFAULT_CSV_PATH


@pytest.fixture
def sample_df():
    """Create small synthetic dataset for cost matrix testing."""
    np.random.seed(42)
    legit = pd.DataFrame({"Amount": np.random.exponential(88, 1000), "Class": 0})
    fraud = pd.DataFrame({"Amount": np.random.exponential(150, 5), "Class": 1})
    return pd.concat([legit, fraud], ignore_index=True)


@pytest.fixture
def real_df():
    """Load real dataset, skip if unavailable."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


class TestCostMatrixDefaults:
    """Test default cost matrix parameter initialization."""

    def test_fp_cost_is_positive(self):
        """FP cost should be positive by default."""
        assert CostMatrix().fp_cost > 0

    def test_fp_cost_is_nontrivial(self):
        """FP cost should be within realistic customer friction bounds."""
        assert 1.0 <= CostMatrix().fp_cost <= 50.0

    def test_tp_cost_is_zero(self):
        """TP cost should be zero."""
        assert CostMatrix().tp_cost == 0.0

    def test_tn_cost_is_zero(self):
        """TN cost should be zero."""
        assert CostMatrix().tn_cost == 0.0


class TestCostMatrixCalibration:
    """Test data-driven calibration of FN cost."""

    def test_calibrate_sets_fn_cost(self, sample_df):
        """Calibrate should populate positive fn_cost."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        assert cm.fn_cost is not None and cm.fn_cost > 0

    def test_fn_cost_matches_avg_fraud(self, sample_df):
        """FN cost should match mean fraud transaction amount."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        expected = sample_df.loc[sample_df["Class"] == 1, "Amount"].mean()
        assert abs(cm.fn_cost - expected) < 0.01

    def test_fn_cost_greater_than_fp_cost(self, sample_df):
        """Missed fraud cost should exceed false alarm cost."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        assert cm.fn_cost > cm.fp_cost

    def test_cost_ratio_positive(self, sample_df):
        """Cost ratio should exceed 1.0."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        assert cm.cost_ratio > 1.0


class TestComputeExpectedCost:
    """Test total business cost computation logic."""

    def test_all_correct_costs_zero(self, sample_df):
        """Zero errors should result in $0 total cost."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=5, fp=0, fn=0, tn=1000)
        assert result["total_cost"] == 0.0

    def test_all_missed_costs_sum(self, sample_df):
        """Missing all frauds should cost fn_cost * n_fraud."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=0, fp=0, fn=5, tn=1000)
        assert abs(result["total_cost"] - 5 * cm.fn_cost) < 0.01

    def test_false_alarms_cost(self, sample_df):
        """False alarms should cost fp_cost * n_fp."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=5, fp=100, fn=0, tn=900)
        assert abs(result["total_cost"] - 100 * cm.fp_cost) < 0.01

    def test_mixed_scenario(self, sample_df):
        """Mixed scenario should sum FN and FP costs accurately."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=3, fp=50, fn=2, tn=950)
        expected = (2 * cm.fn_cost) + (50 * cm.fp_cost)
        assert abs(result["total_cost"] - expected) < 0.01

    def test_result_keys(self, sample_df):
        """Result should contain all expected cost and count keys."""
        cm = CostMatrix().calibrate_from_data(sample_df)
        result = cm.compute_expected_cost(tp=1, fp=1, fn=1, tn=1)
        for key in ["tp_cost", "fp_cost", "fn_cost", "tn_cost", "total_cost", "tp_count", "fp_count", "fn_count", "tn_count"]:
            assert key in result


class TestCostMatrixWithRealData:
    """Tests against real Kaggle dataset."""

    def test_real_fn_cost_reasonable(self, real_df):
        """FN cost should fall within empirical bounds ($50 to $300)."""
        cm = CostMatrix().calibrate_from_data(real_df)
        assert 50 < cm.fn_cost < 300

    def test_real_cost_ratio(self, real_df):
        """Cost ratio should exceed 5:1."""
        cm = CostMatrix().calibrate_from_data(real_df)
        assert cm.cost_ratio > 5
