"""Unit and integration tests for monitoring and orchestration.

Covers:
- Evidently drift detection (positive perturbed test vs. negative unperturbed test)
- Top-5 discriminatory feature anchoring (V17, V14, V12, V10, V16)
- Airflow DAG branching logic and short-circuit evaluation
- DAG evaluation step delegation to evaluate_and_gate_candidate()
"""


from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.features.feature_logic import FEATURE_COLUMNS
from src.monitoring.drift_detector import (
    KEY_DISCRIMINATORY_FEATURES,
    detect_feature_drift,
    generate_synthetic_drift,
    load_reference_features,
)


@pytest.fixture
def synthetic_baseline_data() -> pd.DataFrame:
    """Create a synthetic baseline dataset matching the 30-feature schema."""
    rng = np.random.default_rng(42)
    n_samples = 500
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    return pd.DataFrame(data)


class TestEvidentlyDriftDetector:
    """Test suite for Evidently drift detector."""

    def test_key_discriminatory_features_match_phase1(self):
        """Confirm key discriminatory features are strictly Phase 1 top-5 effect-size features."""
        expected = ["V17", "V14", "V12", "V10", "V16"]
        assert KEY_DISCRIMINATORY_FEATURES == expected

    def test_feature_columns_count(self):
        """Confirm all 30 canonical feature columns are monitored."""
        assert len(FEATURE_COLUMNS) == 30

    def test_unperturbed_negative_case_does_not_flag_drift(self, synthetic_baseline_data):
        """NEGATIVE CASE: Verify that an unperturbed sample from the same distribution does NOT flag drift."""
        rng = np.random.default_rng(999)
        n_samples = 500
        unperturbed_current = pd.DataFrame({
            col: rng.normal(loc=0.0, scale=1.0, size=n_samples)
            for col in FEATURE_COLUMNS
        })

        result = detect_feature_drift(
            reference_data=synthetic_baseline_data,
            current_data=unperturbed_current,
            drift_share_threshold=0.20,
            p_value_threshold=0.05,
        )

        assert result["drift_detected"] is False
        assert result["drift_share"] < 0.20
        assert len(result["key_features_drifted"]) < 2

    def test_perturbed_positive_case_flags_drift(self, synthetic_baseline_data):
        """POSITIVE CASE: Verify that a perturbed sample with shifted features DOES flag drift."""
        perturbed_current = generate_synthetic_drift(
            data=synthetic_baseline_data,
            amount_log_shift=3.0,
            v_mean_shift=2.0,
            v_noise_std=1.0,
            features_to_perturb=["V17", "V14", "V12", "V10"],
            random_state=123,
        )

        result = detect_feature_drift(
            reference_data=synthetic_baseline_data,
            current_data=perturbed_current,
            drift_share_threshold=0.20,
            p_value_threshold=0.05,
        )

        assert result["drift_detected"] is True
        assert len(result["key_features_drifted"]) >= 2
        assert "V17" in result["drifted_features"]
        assert "V14" in result["drifted_features"]

    def test_missing_feature_column_raises_value_error(self, synthetic_baseline_data):
        """Verify error is raised if current or reference data is missing required feature columns."""
        incomplete_df = synthetic_baseline_data.drop(columns=["V17"])
        with pytest.raises(ValueError, match="Missing required feature columns"):
            detect_feature_drift(reference_data=incomplete_df, current_data=synthetic_baseline_data)

    def test_real_feature_store_unperturbed_vs_perturbed(self, tmp_path):
        """Integration test on real feature store snapshot (if available)."""
        feature_store_path = Path("data/feature_store/features_v1.parquet")
        if not feature_store_path.exists():
            pytest.skip("Feature store features_v1.parquet not available.")

        # Load reference (first 2,000 train samples)
        ref_df = load_reference_features(path=feature_store_path, sample_size=2000, random_state=42)
        # Load unperturbed current (independent 2,000 train samples)
        cur_clean = load_reference_features(path=feature_store_path, sample_size=2000, random_state=123)

        # Unperturbed negative test
        clean_res = detect_feature_drift(ref_df, cur_clean)
        assert clean_res["drift_detected"] is False

        # Perturbed positive test
        cur_perturbed = generate_synthetic_drift(cur_clean, amount_log_shift=3.0, v_mean_shift=2.5)
        html_report = tmp_path / "drift_report.html"
        pert_res = detect_feature_drift(ref_df, cur_perturbed, html_report_path=html_report)

        assert pert_res["drift_detected"] is True
        assert pert_res["drift_share"] > 0.20
        assert pert_res["evidently_report_saved"] is True
        assert html_report.exists()


class TestAirflowDAGLogic:
    """Test suite for Airflow DAG branching and task logic."""

    def test_branching_logic_drift_detected_branches_to_retrain(self):
        """Branching test: when drift is detected, branch must select retrain_model_task."""
        from src.orchestration.retrain_dag import decide_branch

        branch = decide_branch(drift_detected=True, is_scheduled_cadence=False)
        assert branch == "retrain_model_task"

    def test_branching_logic_scheduled_cadence_branches_to_retrain(self):
        """Branching test: when on scheduled cadence even without drift, branch must select retrain_model_task."""
        from src.orchestration.retrain_dag import decide_branch

        branch = decide_branch(drift_detected=False, is_scheduled_cadence=True)
        assert branch == "retrain_model_task"

    def test_branching_logic_clean_and_off_schedule_skips_retrain(self):
        """Branching test: when no drift and off-schedule, branch must short-circuit to skip_retrain_task."""
        from src.orchestration.retrain_dag import decide_branch

        branch = decide_branch(drift_detected=False, is_scheduled_cadence=False)
        assert branch == "skip_retrain_task"

    @patch("src.orchestration.retrain_dag.evaluate_and_gate_candidate")
    def test_evaluation_task_calls_phase6_gating_logic(self, mock_gate):
        """Confirm DAG evaluation step calls Phase 6 evaluate_and_gate_candidate() directly."""
        from src.orchestration.retrain_dag import execute_gate_evaluation

        mock_gate.return_value = {
            "action": "promoted",
            "reason": "strictly_superior",
            "promoted_version": "4",
            "candidate_auc_pr": 0.825000,
        }

        mock_client = MagicMock()
        res = execute_gate_evaluation(
            candidate_version="4",
            candidate_auc_pr=0.825000,
            client=mock_client,
            model_name="fraudstream-xgboost",
        )

        mock_gate.assert_called_once_with(
            candidate_version="4",
            candidate_auc_pr=0.825000,
            client=mock_client,
            model_name="fraudstream-xgboost",
            min_improvement=1e-4,
        )
        assert res["action"] == "promoted"
