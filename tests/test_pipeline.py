"""
Tests for the Phase 2 batch pipeline.

Tests cover: data cleaning, feature engineering, train/test split,
scale_pos_weight computation, and cost-optimal threshold selection.

Tests that require the full dataset skip gracefully if it's absent.
Cost-optimal threshold test uses synthetic data with known answers.
"""

import pytest
import pandas as pd
import numpy as np

from src.pipeline.cleaning import (
    validate_no_nulls,
    validate_no_inf,
    transform_amount,
    engineer_time,
    prepare_features,
    get_feature_columns,
    PCA_FEATURES,
)
from src.pipeline.features import (
    build_feature_matrix,
    time_based_split,
    compute_scale_pos_weight,
    split_report,
)
from src.pipeline.evaluate import (
    find_cost_optimal_threshold,
    compute_metrics,
    confusion_at_threshold,
    bootstrap_auc_pr_comparison,
    format_bootstrap_report,
)
from src.eda.cost_matrix import CostMatrix
from src.eda.download_data import DEFAULT_CSV_PATH


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Small synthetic dataset mimicking the real schema."""
    np.random.seed(42)
    n = 200
    data = {"Time": np.sort(np.random.uniform(0, 172800, n))}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    data["Amount"] = np.random.exponential(88, n)
    # 5 frauds at the end (higher time values)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][-5:] = 1
    return pd.DataFrame(data)


@pytest.fixture
def real_df():
    """Load real dataset, skip if absent."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


# ── Cleaning Tests ───────────────────────────────────────────────────


class TestValidation:
    """Tests for data validation functions."""

    def test_validate_no_nulls_passes(self, sample_df):
        """Should pass on clean data."""
        result = validate_no_nulls(sample_df)
        assert result is not None

    def test_validate_no_nulls_fails(self, sample_df):
        """Should raise on data with nulls."""
        df = sample_df.copy()
        df.loc[0, "Amount"] = np.nan
        with pytest.raises(ValueError, match="nulls"):
            validate_no_nulls(df)

    def test_validate_no_inf_passes(self):
        """Should pass on finite data."""
        s = pd.Series([1.0, 2.0, 3.0])
        result = validate_no_inf(s, "test")
        assert len(result) == 3

    def test_validate_no_inf_fails(self):
        """Should raise on data with inf."""
        s = pd.Series([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="infinite"):
            validate_no_inf(s, "test")


class TestAmountTransform:
    """Tests for log1p Amount transformation."""

    def test_log_amount_no_nan(self, sample_df):
        """Transformed Amount should have no NaN values."""
        df = transform_amount(sample_df)
        assert df["log_amount"].isna().sum() == 0

    def test_log_amount_no_inf(self, sample_df):
        """Transformed Amount should have no Inf values."""
        df = transform_amount(sample_df)
        assert np.isinf(df["log_amount"]).sum() == 0

    def test_log_amount_preserves_ordering(self, sample_df):
        """log1p should preserve the ordering of Amount values."""
        df = transform_amount(sample_df)
        # For any two rows, if Amount_a > Amount_b then log_amount_a > log_amount_b
        amounts = df["Amount"].values
        log_amounts = df["log_amount"].values
        for i in range(min(20, len(df) - 1)):
            for j in range(i + 1, min(20, len(df))):
                if amounts[i] > amounts[j]:
                    assert log_amounts[i] > log_amounts[j]

    def test_log_amount_handles_zero(self):
        """log1p(0) should be 0, not NaN or -Inf."""
        df = pd.DataFrame({"Amount": [0.0, 1.0, 100.0]})
        df = transform_amount(df)
        assert df.loc[0, "log_amount"] == 0.0

    def test_real_data_no_nan_inf(self, real_df):
        """Real dataset should produce no NaN/Inf after transform."""
        df = transform_amount(real_df)
        assert df["log_amount"].isna().sum() == 0
        assert np.isinf(df["log_amount"]).sum() == 0


class TestTimeEngineering:
    """Tests for hour_of_day extraction."""

    def test_hour_range(self, sample_df):
        """hour_of_day should be in [0, 24)."""
        df = engineer_time(sample_df)
        assert df["hour_of_day"].min() >= 0.0
        assert df["hour_of_day"].max() < 24.0

    def test_hour_cyclic(self):
        """Time = 90000s (25 hours) should map to hour 1."""
        df = pd.DataFrame({"Time": [90000.0]})
        df = engineer_time(df)
        assert abs(df.loc[0, "hour_of_day"] - 1.0) < 0.01


class TestFeatureColumns:
    """Tests for feature column selection."""

    def test_feature_columns_count(self):
        """Should have 30 features: 28 PCA + log_amount + hour_of_day."""
        cols = get_feature_columns()
        assert len(cols) == 30

    def test_no_raw_time_or_amount(self):
        """Feature columns should NOT include raw Time or Amount."""
        cols = get_feature_columns()
        assert "Time" not in cols
        assert "Amount" not in cols

    def test_no_class_label(self):
        """Feature columns should NOT include the target Class."""
        cols = get_feature_columns()
        assert "Class" not in cols

    def test_prepare_features_produces_all_columns(self, sample_df):
        """prepare_features should create all expected feature columns."""
        df = prepare_features(sample_df)
        cols = get_feature_columns()
        for col in cols:
            assert col in df.columns, f"Missing column: {col}"


# ── Split Tests ──────────────────────────────────────────────────────


class TestTimeBasedSplit:
    """Tests for time-based train/test split."""

    def test_train_before_test(self, sample_df):
        """All train transactions should be earlier than test transactions."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        assert train_df["Time"].max() <= test_df["Time"].min()

    def test_split_sizes(self, sample_df):
        """Train should be ~80%, test ~20%."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        total = len(train_df) + len(test_df)
        assert abs(len(train_df) / total - 0.8) < 0.01

    def test_no_overlap(self, sample_df):
        """Train and test should not share any rows."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        assert len(train_df) + len(test_df) == len(sample_df)

    def test_fraud_rate_approximately_preserved(self, real_df):
        """Both sets should have a fraud rate in a reasonable range."""
        train_df, test_df = time_based_split(real_df, 0.8)
        train_rate = (train_df["Class"] == 1).mean()
        test_rate = (test_df["Class"] == 1).mean()
        # Time-based split won't exactly match, but should be in
        # the same order of magnitude
        assert 0.05 < train_rate * 100 < 0.5, f"Train fraud rate {train_rate:.4%} out of range"
        assert 0.05 < test_rate * 100 < 0.5, f"Test fraud rate {test_rate:.4%} out of range"


class TestScalePosWeight:
    """Tests for scale_pos_weight computation."""

    def test_not_hardcoded(self):
        """scale_pos_weight should be computed from data, not hardcoded."""
        # Create synthetic data with fraud distributed across time
        # (not just at the end, to ensure some fraud in training split)
        np.random.seed(42)
        n = 200
        data = {"Time": np.sort(np.random.uniform(0, 172800, n))}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n)
        data["Amount"] = np.random.exponential(88, n)
        data["Class"] = np.zeros(n, dtype=int)
        # Place fraud in first half and second half
        data["Class"][10] = 1
        data["Class"][30] = 1
        data["Class"][50] = 1
        data["Class"][180] = 1
        data["Class"][190] = 1
        df = pd.DataFrame(data)

        train_df, _ = time_based_split(df, 0.8)
        _, y_train = build_feature_matrix(train_df)
        spw = compute_scale_pos_weight(y_train)
        n_fraud = int((y_train == 1).sum())
        n_legit = int((y_train == 0).sum())
        assert n_fraud > 0, "Need fraud in training set for this test"
        assert spw == n_legit / n_fraud

    def test_real_data_reasonable(self, real_df):
        """Real data should give scale_pos_weight in range 400-700."""
        train_df, _ = time_based_split(real_df, 0.8)
        _, y_train = build_feature_matrix(train_df)
        spw = compute_scale_pos_weight(y_train)
        assert 400 < spw < 700, f"scale_pos_weight {spw:.1f} outside expected range"

    def test_error_on_no_fraud(self):
        """Should raise if no fraud in training set."""
        y = pd.Series([0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="No fraud"):
            compute_scale_pos_weight(y)


# ── Cost-Optimal Threshold Tests ─────────────────────────────────────


class TestCostOptimalThreshold:
    """Tests for cost-based threshold selection.

    Uses synthetic data with known answers to verify the minimum-cost
    threshold is correctly identified — not just "runs without error."
    """

    def test_known_answer_threshold(self):
        """On synthetic data, the cost-optimal threshold should minimize cost.

        Setup: 100 samples. 10 are fraud (Class=1) with probabilities
        clustered around 0.8. 90 are legit (Class=0) with probabilities
        clustered around 0.1. A threshold around 0.5 should correctly
        separate them. Setting threshold too low (0.01) catches all fraud
        but creates many FP. Setting too high (0.99) misses fraud.
        """
        np.random.seed(42)
        n_legit = 90
        n_fraud = 10

        y_true = np.array([0] * n_legit + [1] * n_fraud)
        # Legit probs mostly low, fraud probs mostly high
        prob_legit = np.clip(np.random.normal(0.1, 0.05, n_legit), 0, 1)
        prob_fraud = np.clip(np.random.normal(0.8, 0.1, n_fraud), 0, 1)
        y_prob = np.concatenate([prob_legit, prob_fraud])

        cm = CostMatrix(fn_cost=122.21, fp_cost=10.0)
        result = find_cost_optimal_threshold(y_true, y_prob, cm)

        # The optimal threshold should be somewhere between 0.15 and 0.8
        assert 0.05 < result["optimal_threshold"] < 0.9, (
            f"Optimal threshold {result['optimal_threshold']:.4f} outside expected range"
        )

        # The minimum cost should be less than the cost at extreme thresholds
        # At t=0 (predict all fraud): FP_cost = 90 * $10 = $900
        # At t=1 (predict none fraud): FN_cost = 10 * $122.21 = $1222.10
        assert result["min_cost"] < 900.0, (
            f"Min cost ${result['min_cost']:.2f} should be less than all-positive cost"
        )
        assert result["min_cost"] < 1222.0, (
            f"Min cost ${result['min_cost']:.2f} should be less than all-negative cost"
        )

    def test_threshold_minimizes_cost(self):
        """Verify the returned threshold actually has the minimum cost.

        Compare the cost at the optimal threshold against costs at several
        other thresholds — it should be lower or equal to all of them.
        """
        np.random.seed(123)
        y_true = np.array([0] * 50 + [1] * 5)
        y_prob = np.concatenate([
            np.random.uniform(0, 0.4, 50),
            np.random.uniform(0.5, 1.0, 5),
        ])

        cm = CostMatrix(fn_cost=100.0, fp_cost=5.0)
        result = find_cost_optimal_threshold(y_true, y_prob, cm, n_thresholds=500)

        # Check cost at other thresholds
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            other = confusion_at_threshold(y_true, y_prob, t, cm)
            assert result["min_cost"] <= other["total_cost"] + 0.01, (
                f"Optimal cost ${result['min_cost']:.2f} > cost at t={t}: ${other['total_cost']:.2f}"
            )

    def test_high_cost_ratio_favors_lower_threshold(self):
        """When FN cost >> FP cost, optimal threshold should be lower.

        With extreme cost ratio (FN=1000, FP=1), the model should be very
        aggressive (low threshold) to avoid missing any fraud.
        """
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 5)
        y_prob = np.concatenate([
            np.random.uniform(0, 0.5, 100),
            np.random.uniform(0.3, 0.9, 5),
        ])

        # Very high FN cost
        cm_high = CostMatrix(fn_cost=1000.0, fp_cost=1.0)
        result_high = find_cost_optimal_threshold(y_true, y_prob, cm_high)

        # Lower FN cost
        cm_low = CostMatrix(fn_cost=10.0, fp_cost=10.0)
        result_low = find_cost_optimal_threshold(y_true, y_prob, cm_low)

        # Higher cost ratio should produce a lower (more aggressive) threshold
        assert result_high["optimal_threshold"] <= result_low["optimal_threshold"], (
            f"High cost ratio threshold ({result_high['optimal_threshold']:.4f}) "
            f"should be <= low ratio ({result_low['optimal_threshold']:.4f})"
        )


class TestBootstrapModelComparison:
    """Tests for bootstrap AUC-PR comparison between models."""

    def test_bootstrap_output_structure(self):
        """Verify bootstrap returns expected dictionary keys and lengths."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 10)
        p_a = np.random.uniform(0, 1, 110)
        p_b = np.random.uniform(0, 1, 110)

        res = bootstrap_auc_pr_comparison(
            y_true, p_a, p_b, n_bootstraps=50, random_state=42
        )

        assert "model_a" in res
        assert "model_b" in res
        assert "diff" in res
        assert "prob_a_superior" in res
        assert "prob_b_superior" in res
        assert "statistically_significant" in res
        assert len(res["raw_scores_a"]) == 50
        assert len(res["raw_scores_b"]) == 50
        assert len(res["raw_diffs"]) == 50
        assert res["model_a"]["ci_lower"] <= res["model_a"]["ci_upper"]
        assert res["model_b"]["ci_lower"] <= res["model_b"]["ci_upper"]
        assert res["diff"]["ci_lower"] <= res["diff"]["ci_upper"]

    def test_identical_models_yield_zero_difference(self):
        """When two models produce identical probabilities, diff must be zero."""
        np.random.seed(42)
        y_true = np.array([0] * 80 + [1] * 10)
        p_a = np.random.uniform(0, 1, 90)

        res = bootstrap_auc_pr_comparison(
            y_true, p_a, p_a, n_bootstraps=50, random_state=42
        )

        assert abs(res["diff"]["mean"]) < 1e-9
        assert abs(res["diff"]["ci_lower"]) < 1e-9
        assert abs(res["diff"]["ci_upper"]) < 1e-9
        assert not res["statistically_significant"]

    def test_strictly_superior_model_detected(self):
        """When Model A has near-perfect scores and Model B is random, A should dominate."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 15)
        # Model A assigns 0.99 to all frauds and 0.01 to legits
        p_a = np.where(y_true == 1, 0.99, 0.01)
        # Model B is random noise
        p_b = np.random.uniform(0, 1, 115)

        res = bootstrap_auc_pr_comparison(
            y_true, p_a, p_b, n_bootstraps=100, random_state=42
        )

        assert res["diff"]["ci_lower"] > 0
        assert res["statistically_significant"] is True
        assert res["prob_a_superior"] == 1.0

    def test_format_bootstrap_report(self):
        """Verify report string formatting contains expected headers and numbers."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 10)
        p_a = np.random.uniform(0, 1, 110)
        p_b = np.random.uniform(0, 1, 110)

        res = bootstrap_auc_pr_comparison(
            y_true, p_a, p_b, n_bootstraps=30, random_state=42
        )
        report = format_bootstrap_report(res)

        assert "BOOTSTRAP MODEL COMPARISON REPORT" in report
        assert "95% Bootstrap CI" in report
        assert "Paired Difference" in report
