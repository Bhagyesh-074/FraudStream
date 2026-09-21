"""Tests for data cleaning, features, split, and evaluation."""

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


@pytest.fixture
def sample_df():
    """Synthetic dataset with 200 rows mimicking creditcard.csv schema."""
    np.random.seed(42)
    n = 200
    data = {"Time": np.sort(np.random.uniform(0, 172800, n))}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    data["Amount"] = np.random.exponential(88, n)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][-5:] = 1
    return pd.DataFrame(data)


@pytest.fixture
def real_df():
    """Load real dataset, skip if absent."""
    if not DEFAULT_CSV_PATH.is_file():
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return pd.read_csv(DEFAULT_CSV_PATH)


class TestValidation:
    """Tests for dataset validation checks."""

    def test_validate_no_nulls_passes(self, sample_df):
        """Clean dataset should pass null validation."""
        assert validate_no_nulls(sample_df) is not None

    def test_validate_no_nulls_fails(self, sample_df):
        """Dataset with nulls should raise ValueError."""
        df = sample_df.copy()
        df.loc[0, "Amount"] = np.nan
        with pytest.raises(ValueError, match="nulls"):
            validate_no_nulls(df)

    def test_validate_no_inf_passes(self):
        """Finite values should pass inf validation."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert len(validate_no_inf(s, "test")) == 3

    def test_validate_no_inf_fails(self):
        """Infinite values should raise ValueError."""
        s = pd.Series([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="infinite"):
            validate_no_inf(s, "test")


class TestAmountTransform:
    """Tests for log1p Amount transformation."""

    def test_log_amount_no_nan(self, sample_df):
        """Transformed column should contain zero NaNs."""
        df = transform_amount(sample_df)
        assert df["log_amount"].isna().sum() == 0

    def test_log_amount_no_inf(self, sample_df):
        """Transformed column should contain zero Infs."""
        df = transform_amount(sample_df)
        assert np.isinf(df["log_amount"]).sum() == 0

    def test_log_amount_preserves_ordering(self, sample_df):
        """log1p should preserve monotonicity of transaction amounts."""
        df = transform_amount(sample_df)
        amounts = df["Amount"].values
        log_amounts = df["log_amount"].values
        for i in range(min(20, len(df) - 1)):
            for j in range(i + 1, min(20, len(df))):
                if amounts[i] > amounts[j]:
                    assert log_amounts[i] > log_amounts[j]

    def test_log_amount_handles_zero(self):
        """log1p(0) should equal 0.0."""
        df = pd.DataFrame({"Amount": [0.0, 1.0, 100.0]})
        assert transform_amount(df).loc[0, "log_amount"] == 0.0

    def test_real_data_no_nan_inf(self, real_df):
        """Real dataset should produce zero NaNs or Infs."""
        df = transform_amount(real_df)
        assert df["log_amount"].isna().sum() == 0 and np.isinf(df["log_amount"]).sum() == 0


class TestTimeEngineering:
    """Tests for cyclic hour_of_day feature extraction."""

    def test_hour_range(self, sample_df):
        """hour_of_day should stay within [0, 24)."""
        df = engineer_time(sample_df)
        assert 0.0 <= df["hour_of_day"].min() and df["hour_of_day"].max() < 24.0

    def test_hour_cyclic(self):
        """Time = 90000s (25h) should map to hour 1.0."""
        df = pd.DataFrame({"Time": [90000.0]})
        assert abs(engineer_time(df).loc[0, "hour_of_day"] - 1.0) < 0.01


class TestFeatureColumns:
    """Tests for feature matrix columns."""

    def test_feature_columns_count(self):
        """Should contain exactly 30 features."""
        assert len(get_feature_columns()) == 30

    def test_no_raw_time_or_amount(self):
        """Raw Time and Amount must be excluded."""
        cols = get_feature_columns()
        assert "Time" not in cols and "Amount" not in cols

    def test_no_class_label(self):
        """Target Class label must be excluded from features."""
        assert "Class" not in get_feature_columns()

    def test_prepare_features_produces_all_columns(self, sample_df):
        """prepare_features should produce all required feature columns."""
        df = prepare_features(sample_df)
        for col in get_feature_columns():
            assert col in df.columns


class TestTimeBasedSplit:
    """Tests for chronological train/test splitting."""

    def test_train_before_test(self, sample_df):
        """Train timestamps must precede test timestamps."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        assert train_df["Time"].max() <= test_df["Time"].min()

    def test_split_sizes(self, sample_df):
        """Split sizes should match 80% train and 20% test."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        assert abs(len(train_df) / (len(train_df) + len(test_df)) - 0.8) < 0.01

    def test_no_overlap(self, sample_df):
        """Train and test splits should have zero row overlap."""
        train_df, test_df = time_based_split(sample_df, 0.8)
        assert len(train_df) + len(test_df) == len(sample_df)

    def test_fraud_rate_approximately_preserved(self, real_df):
        """Fraud rates should remain bounded in both splits."""
        train_df, test_df = time_based_split(real_df, 0.8)
        assert 0.05 < (train_df["Class"] == 1).mean() * 100 < 0.5
        assert 0.05 < (test_df["Class"] == 1).mean() * 100 < 0.5


class TestScalePosWeight:
    """Tests for dynamic scale_pos_weight computation."""

    def test_not_hardcoded(self):
        """scale_pos_weight should match legit/fraud count ratio in train split."""
        np.random.seed(42)
        n = 200
        data = {"Time": np.sort(np.random.uniform(0, 172800, n))}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n)
        data["Amount"] = np.random.exponential(88, n)
        data["Class"] = np.zeros(n, dtype=int)
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
        assert spw == n_legit / n_fraud

    def test_real_data_reasonable(self, real_df):
        """scale_pos_weight on real data should be between 400 and 700."""
        train_df, _ = time_based_split(real_df, 0.8)
        _, y_train = build_feature_matrix(train_df)
        assert 400 < compute_scale_pos_weight(y_train) < 700

    def test_error_on_no_fraud(self):
        """Should raise ValueError if training set has zero fraud."""
        with pytest.raises(ValueError, match="No fraud"):
            compute_scale_pos_weight(pd.Series([0, 0, 0]))


class TestCostOptimalThreshold:
    """Tests for business cost-optimal threshold search."""

    def test_known_answer_threshold(self):
        """Cost-optimal threshold should separate synthetic bimodal probabilities."""
        np.random.seed(42)
        y_true = np.array([0] * 90 + [1] * 10)
        prob_legit = np.clip(np.random.normal(0.1, 0.05, 90), 0, 1)
        prob_fraud = np.clip(np.random.normal(0.8, 0.1, 10), 0, 1)
        y_prob = np.concatenate([prob_legit, prob_fraud])

        cm = CostMatrix(fn_cost=122.21, fp_cost=10.0)
        result = find_cost_optimal_threshold(y_true, y_prob, cm)
        assert 0.05 < result["optimal_threshold"] < 0.9
        assert result["min_cost"] < 900.0 and result["min_cost"] < 1222.0

    def test_threshold_minimizes_cost(self):
        """Selected threshold must produce minimum cost compared to arbitrary cutoffs."""
        np.random.seed(123)
        y_true = np.array([0] * 50 + [1] * 5)
        y_prob = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.5, 1.0, 5)])

        cm = CostMatrix(fn_cost=100.0, fp_cost=5.0)
        result = find_cost_optimal_threshold(y_true, y_prob, cm, n_thresholds=500)
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            other = confusion_at_threshold(y_true, y_prob, t, cm)
            assert result["min_cost"] <= other["total_cost"] + 0.01

    def test_high_cost_ratio_favors_lower_threshold(self):
        """Higher FN/FP cost ratio should yield a lower (more aggressive) threshold."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 5)
        y_prob = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.3, 0.9, 5)])

        cm_high = CostMatrix(fn_cost=1000.0, fp_cost=1.0)
        cm_low = CostMatrix(fn_cost=10.0, fp_cost=10.0)
        assert find_cost_optimal_threshold(y_true, y_prob, cm_high)["optimal_threshold"] <= find_cost_optimal_threshold(y_true, y_prob, cm_low)["optimal_threshold"]


class TestBootstrapModelComparison:
    """Tests for paired bootstrap model comparison."""

    def test_bootstrap_output_structure(self):
        """Verify output dictionary keys, lengths, and valid CI bounds."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 10)
        res = bootstrap_auc_pr_comparison(
            y_true, np.random.uniform(0, 1, 110), np.random.uniform(0, 1, 110), n_bootstraps=50, random_state=42
        )
        for key in ["model_a", "model_b", "diff", "prob_a_superior", "prob_b_superior", "statistically_significant"]:
            assert key in res
        assert len(res["raw_scores_a"]) == 50 and len(res["raw_scores_b"]) == 50 and len(res["raw_diffs"]) == 50
        assert res["model_a"]["ci_lower"] <= res["model_a"]["ci_upper"]
        assert res["diff"]["ci_lower"] <= res["diff"]["ci_upper"]

    def test_identical_models_yield_zero_difference(self):
        """Identical predictions must yield zero difference and no significance."""
        np.random.seed(42)
        y_true = np.array([0] * 80 + [1] * 10)
        p = np.random.uniform(0, 1, 90)
        res = bootstrap_auc_pr_comparison(y_true, p, p, n_bootstraps=50, random_state=42)
        assert abs(res["diff"]["mean"]) < 1e-9 and not res["statistically_significant"]

    def test_strictly_superior_model_detected(self):
        """Near-perfect model vs random guessing must achieve statistical significance."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 15)
        p_a = np.where(y_true == 1, 0.99, 0.01)
        res = bootstrap_auc_pr_comparison(y_true, p_a, np.random.uniform(0, 1, 115), n_bootstraps=100, random_state=42)
        assert res["diff"]["ci_lower"] > 0 and res["statistically_significant"] and res["prob_a_superior"] == 1.0

    def test_format_bootstrap_report(self):
        """Verify formatted bootstrap report contains key diagnostic terms."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 10)
        res = bootstrap_auc_pr_comparison(
            y_true, np.random.uniform(0, 1, 110), np.random.uniform(0, 1, 110), n_bootstraps=30, random_state=42
        )
        report = format_bootstrap_report(res)
        assert "BOOTSTRAP MODEL COMPARISON REPORT" in report and "95% Bootstrap CI" in report
