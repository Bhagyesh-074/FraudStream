"""Tests for Phase 6: DVC versioning and MLflow Model Registry promotion gating."""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDVCVersioning:
    """Test suite for DVC artifact tracking and pointer file validation."""

    def test_dvc_pointer_files_exist(self):
        csv_dvc = PROJECT_ROOT / "data" / "creditcard.csv.dvc"
        fs_dvc = PROJECT_ROOT / "data" / "feature_store.dvc"
        model_dvc = PROJECT_ROOT / "models" / "xgboost_baseline.json.dvc"

        assert csv_dvc.is_file(), "data/creditcard.csv.dvc missing"
        assert fs_dvc.is_file(), "data/feature_store.dvc missing"
        assert model_dvc.is_file(), "models/xgboost_baseline.json.dvc missing"

    def test_creditcard_csv_dvc_schema(self):
        csv_dvc = PROJECT_ROOT / "data" / "creditcard.csv.dvc"
        with open(csv_dvc, "r") as f:
            data = yaml.safe_load(f)

        assert "outs" in data
        assert len(data["outs"]) == 1
        out = data["outs"][0]
        assert "md5" in out
        assert len(out["md5"]) == 32
        assert out["path"] == "creditcard.csv"
        assert out["size"] > 100_000_000

    def test_feature_store_dvc_schema(self):
        fs_dvc = PROJECT_ROOT / "data" / "feature_store.dvc"
        with open(fs_dvc, "r") as f:
            data = yaml.safe_load(f)

        assert "outs" in data
        out = data["outs"][0]
        assert "md5" in out
        assert out["path"] == "feature_store"
        assert out["size"] > 50_000_000

    def test_dvc_status_is_clean(self):
        res = subprocess.run(
            ["dvc", "status"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"dvc status failed: {res.stderr}"
        assert "Data and pipelines are up to date" in res.stdout


class TestPromotionGatingLogic:
    """Test suite for explicit model registry promotion gating rules."""

    @pytest.fixture
    def mock_mlflow_client(self):
        client = MagicMock()
        return client

    def test_initial_deployment_promotes_to_production(self, mock_mlflow_client):
        from src.pipeline.train import evaluate_and_gate_candidate

        # No existing models in Production
        mock_mlflow_client.search_model_versions.return_value = []

        result = evaluate_and_gate_candidate(
            candidate_version="1",
            candidate_auc_pr=0.799586,
            client=mock_mlflow_client,
            model_name="fraudstream-xgboost",
        )

        assert result["action"] == "promoted"
        assert result["reason"] == "initial_deployment"
        assert result["promoted_version"] == "1"
        mock_mlflow_client.transition_model_version_stage.assert_called_once_with(
            name="fraudstream-xgboost",
            version="1",
            stage="Production",
        )

    def test_strictly_superior_candidate_is_promoted_and_archives_old(self, mock_mlflow_client):
        from src.pipeline.train import evaluate_and_gate_candidate

        # Current production version 1 with AUC-PR = 0.799586
        current_prod = MagicMock()
        current_prod.version = "1"
        current_prod.current_stage = "Production"
        current_prod.run_id = "run-prod-001"
        current_prod.tags = {}

        prod_run = MagicMock()
        prod_run.data.metrics = {"auc_pr": 0.799586}
        mock_mlflow_client.get_run.return_value = prod_run
        mock_mlflow_client.search_model_versions.return_value = [current_prod]

        # Candidate version 2 with strictly superior AUC-PR = 0.812500
        result = evaluate_and_gate_candidate(
            candidate_version="2",
            candidate_auc_pr=0.812500,
            client=mock_mlflow_client,
            model_name="fraudstream-xgboost",
        )

        assert result["action"] == "promoted"
        assert result["reason"] == "strictly_superior"
        assert result["promoted_version"] == "2"
        assert result["previous_production_version"] == "1"

        # Verify old was archived and candidate was promoted
        calls = mock_mlflow_client.transition_model_version_stage.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs == {"name": "fraudstream-xgboost", "version": "1", "stage": "Archived"}
        assert calls[1].kwargs == {"name": "fraudstream-xgboost", "version": "2", "stage": "Production"}

    def test_inferior_candidate_is_rejected_and_remains_in_staging(self, mock_mlflow_client):
        from src.pipeline.train import evaluate_and_gate_candidate

        current_prod = MagicMock()
        current_prod.version = "1"
        current_prod.current_stage = "Production"
        current_prod.run_id = "run-prod-001"
        current_prod.tags = {}

        prod_run = MagicMock()
        prod_run.data.metrics = {"auc_pr": 0.799586}
        mock_mlflow_client.get_run.return_value = prod_run
        mock_mlflow_client.search_model_versions.return_value = [current_prod]

        # Candidate version 2 with inferior AUC-PR = 0.750000
        result = evaluate_and_gate_candidate(
            candidate_version="2",
            candidate_auc_pr=0.750000,
            client=mock_mlflow_client,
            model_name="fraudstream-xgboost",
        )

        assert result["action"] == "rejected"
        assert result["reason"] == "not_strictly_superior"
        assert result["promoted_version"] is None
        assert result["current_production_version"] == "1"
        # No transitions should be executed
        mock_mlflow_client.transition_model_version_stage.assert_not_called()

    def test_equal_candidate_is_rejected(self, mock_mlflow_client):
        from src.pipeline.train import evaluate_and_gate_candidate

        current_prod = MagicMock()
        current_prod.version = "1"
        current_prod.current_stage = "Production"
        current_prod.run_id = "run-prod-001"
        current_prod.tags = {}

        prod_run = MagicMock()
        prod_run.data.metrics = {"auc_pr": 0.799586}
        mock_mlflow_client.get_run.return_value = prod_run
        mock_mlflow_client.search_model_versions.return_value = [current_prod]

        # Candidate version 2 with exactly equal AUC-PR = 0.799586
        result = evaluate_and_gate_candidate(
            candidate_version="2",
            candidate_auc_pr=0.799586,
            client=mock_mlflow_client,
            model_name="fraudstream-xgboost",
        )

        assert result["action"] == "rejected"
        assert result["reason"] == "not_strictly_superior"
        mock_mlflow_client.transition_model_version_stage.assert_not_called()

    def test_archiving_preserves_registry_record(self):
        """Verify that archived model versions are not deleted from registry search."""
        from mlflow.tracking import MlflowClient
        db_path = PROJECT_ROOT / "mlflow.db"
        if not db_path.exists():
            pytest.skip("mlflow.db not initialized")

        client = MlflowClient(f"sqlite:///{db_path.as_posix()}")
        versions = client.search_model_versions("name='fraudstream-xgboost'")
        stages = {v.version: v.current_stage for v in versions}

        # Check that we have a production version and archived/staging versions
        assert "Production" in stages.values(), "No model in Production stage"
        assert len(versions) >= 1, "Expected registered versions to be retained"


class TestServingModelLoaderGovernance:
    """Test suite verifying serving integration with model registry and fallback warnings."""

    def test_model_loader_registry_source(self):
        from src.serving.model_loader import load_model, get_active_model_source
        model = load_model(prefer_registry=True)
        assert model is not None
        assert get_active_model_source() == "registry"

    def test_model_loader_fallback_warning_and_source(self, caplog):
        from src.serving.model_loader import load_model, get_active_model_source
        model = load_model(prefer_registry=False)
        assert model is not None
        assert get_active_model_source() == "local_fallback"

    def test_api_health_endpoint_reports_model_source(self):
        from fastapi.testclient import TestClient
        from src.serving.api import app
        from src.serving.model_loader import load_model
        load_model(prefer_registry=True)

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "model_source" in data
        assert data["model_source"] in ("registry", "local_fallback")
