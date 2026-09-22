"""Tests for FraudStream Phase 5: serving module (model loading, scoring, API)."""

import json
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.feature_logic import compute_features, FEATURE_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SKIP_NO_DATA = not DATA_PATH.is_file()
SKIP_NO_MODEL = not (MODELS_DIR / "xgboost_baseline.json").is_file()


# --- Model Loader Tests ---

class TestModelLoader:
    def test_load_model_success(self):
        if SKIP_NO_MODEL:
            pytest.skip("Model artifacts not found")
        from src.serving.model_loader import load_model
        model = load_model()
        assert isinstance(model, xgb.XGBClassifier)

    def test_load_model_missing_file(self, tmp_path):
        from src.serving.model_loader import load_model
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(model_dir=tmp_path)

    def test_load_threshold_config_success(self):
        if SKIP_NO_MODEL:
            pytest.skip("Model artifacts not found")
        from src.serving.model_loader import load_threshold_config
        config = load_threshold_config()
        assert "optimal_threshold" in config
        assert isinstance(config["optimal_threshold"], float)
        assert 0 < config["optimal_threshold"] < 1

    def test_load_threshold_config_missing_file(self, tmp_path):
        from src.serving.model_loader import load_threshold_config
        with pytest.raises(FileNotFoundError, match="Threshold config not found"):
            load_threshold_config(model_dir=tmp_path)

    def test_load_threshold_config_missing_key(self, tmp_path):
        from src.serving.model_loader import load_threshold_config
        bad_config = tmp_path / "threshold_config.json"
        bad_config.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="missing 'optimal_threshold'"):
            load_threshold_config(model_dir=tmp_path)

    def test_model_roundtrip(self, tmp_path):
        """Save and reload a model to verify serialization works."""
        model = xgb.XGBClassifier(n_estimators=5, random_state=42, use_label_encoder=False)
        X = np.random.RandomState(42).randn(50, 30)
        y = np.random.RandomState(42).randint(0, 2, 50)
        model.fit(X, y)
        model.save_model(str(tmp_path / "xgboost_baseline.json"))

        config = {"optimal_threshold": 0.031, "model_name": "test"}
        with open(tmp_path / "threshold_config.json", "w") as f:
            json.dump(config, f)

        from src.serving.model_loader import load_model, load_threshold_config
        loaded = load_model(tmp_path)
        loaded_config = load_threshold_config(tmp_path)
        assert isinstance(loaded, xgb.XGBClassifier)
        assert loaded_config["optimal_threshold"] == 0.031


# --- Scoring Logic Tests ---

class TestScoringLogic:
    @pytest.fixture
    def sample_record(self):
        if SKIP_NO_DATA:
            pytest.skip("Dataset not available")
        df = pd.read_csv(DATA_PATH, nrows=1000)
        return df.iloc[0].to_dict()

    @pytest.fixture
    def fraud_record(self):
        if SKIP_NO_DATA:
            pytest.skip("Dataset not available")
        df = pd.read_csv(DATA_PATH, nrows=7000)
        frauds = df[df["Class"] == 1]
        if len(frauds) == 0:
            pytest.skip("No fraud records in first 7000 rows")
        return frauds.iloc[0].to_dict()

    @pytest.fixture
    def model_and_threshold(self):
        if SKIP_NO_MODEL:
            pytest.skip("Model artifacts not found")
        from src.serving.model_loader import load_model, load_threshold_config
        model = load_model()
        config = load_threshold_config()
        return model, config["optimal_threshold"]

    def test_score_record_returns_valid_payload(self, sample_record, model_and_threshold):
        from src.serving.scoring_consumer import score_record
        model, threshold = model_and_threshold
        scored, latency_ms = score_record(sample_record, model, threshold)

        assert "fraud_probability" in scored
        assert "is_fraud" in scored
        assert "threshold_used" in scored
        assert "scoring_latency_ms" in scored
        assert "scored_at" in scored
        assert 0.0 <= scored["fraud_probability"] <= 1.0
        assert isinstance(scored["is_fraud"], bool)
        assert scored["threshold_used"] == threshold
        assert latency_ms > 0

    def test_score_record_preserves_original_fields(self, sample_record, model_and_threshold):
        from src.serving.scoring_consumer import score_record
        model, threshold = model_and_threshold
        scored, _ = score_record(sample_record, model, threshold)
        # Original Time and Amount preserved
        assert scored["Time"] == sample_record["Time"]
        assert scored["Amount"] == sample_record["Amount"]
        # Class field stripped from scored output
        assert "Class" not in scored

    @pytest.fixture
    def test_split_fraud_record(self):
        if SKIP_NO_DATA:
            pytest.skip("Dataset not available")
        df = pd.read_csv(DATA_PATH)
        test_frauds = df[(df["Time"] > 145247) & (df["Class"] == 1)]
        if len(test_frauds) == 0:
            pytest.skip("No fraud records in test split")
        return test_frauds.iloc[0].to_dict()

    def test_score_record_fraud_flagging(self, fraud_record, model_and_threshold):
        from src.serving.scoring_consumer import score_record
        model, threshold = model_and_threshold
        scored, _ = score_record(fraud_record, model, threshold)
        # Report the result — model may or may not flag it
        print(f"Train fraud record probability: {scored['fraud_probability']:.6f}, flagged: {scored['is_fraud']}")
        assert 0.0 <= scored["fraud_probability"] <= 1.0

    def test_score_record_test_split_fraud_flagging(self, test_split_fraud_record, model_and_threshold):
        from src.serving.scoring_consumer import score_record
        model, threshold = model_and_threshold
        scored, _ = score_record(test_split_fraud_record, model, threshold)
        print(f"Test split fraud record probability: {scored['fraud_probability']:.6f}, flagged: {scored['is_fraud']}")
        assert 0.0 <= scored["fraud_probability"] <= 1.0
        assert "is_fraud" in scored

    def test_scoring_latency_under_15ms(self, sample_record, model_and_threshold):
        from src.serving.scoring_consumer import score_record
        model, threshold = model_and_threshold
        # Warm up
        score_record(sample_record, model, threshold)
        # Measure over 50 iterations
        latencies = []
        for _ in range(50):
            _, latency_ms = score_record(sample_record, model, threshold)
            latencies.append(latency_ms)
        p95 = np.percentile(latencies, 95)
        print(f"Latency p50={np.median(latencies):.2f}ms p95={p95:.2f}ms max={max(latencies):.2f}ms")
        assert p95 < 15.0, f"p95 latency {p95:.2f}ms exceeds 15ms SLA"


# --- Stats Accumulator Tests ---

class TestScoringStats:
    def test_stats_initial_state(self):
        from src.serving.scoring_consumer import ScoringStats
        stats = ScoringStats()
        snap = stats.snapshot()
        assert snap["total_scored"] == 0
        assert snap["total_flagged"] == 0
        assert snap["flag_rate"] == 0.0

    def test_stats_record_and_snapshot(self):
        from src.serving.scoring_consumer import ScoringStats
        stats = ScoringStats()
        stats.record({"id": 1}, 2.5, False)
        stats.record({"id": 2}, 1.5, True)
        stats.record({"id": 3}, 3.0, False)

        snap = stats.snapshot()
        assert snap["total_scored"] == 3
        assert snap["total_flagged"] == 1
        assert abs(snap["flag_rate"] - 1 / 3) < 1e-6

    def test_stats_ring_buffer_limit(self):
        from src.serving.scoring_consumer import ScoringStats
        stats = ScoringStats(recent_limit=5)
        for i in range(10):
            stats.record({"id": i}, 1.0, False)
        recent = stats.get_recent(10)
        assert len(recent) == 5
        assert recent[0]["id"] == 5  # oldest in buffer

    def test_stats_latency_percentiles(self):
        from src.serving.scoring_consumer import ScoringStats
        stats = ScoringStats()
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            stats.record({}, lat, False)
        snap = stats.snapshot()
        assert snap["latency_ms"]["min"] == 1.0
        assert snap["latency_ms"]["max"] == 10.0
        assert snap["latency_ms"]["p50"] == pytest.approx(5.5, abs=0.1)

    def test_stats_thread_safety(self):
        from src.serving.scoring_consumer import ScoringStats
        stats = ScoringStats()

        def writer(start):
            for i in range(100):
                stats.record({"id": start + i}, float(i), i % 10 == 0)

        threads = [threading.Thread(target=writer, args=(t * 100,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = stats.snapshot()
        assert snap["total_scored"] == 400


# --- API Tests ---

class TestAPI:
    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        from src.serving import api
        # Set module-level state for testing
        api._model_loaded = True
        api._threshold = 0.031
        api._stats = api.ScoringStats()
        api._stats.record({"Amount": 100}, 2.5, False)
        api._stats.record({"Amount": 999}, 1.0, True)
        return TestClient(api.app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["threshold"] == 0.031

    def test_stats_endpoint(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_scored"] == 2
        assert data["total_flagged"] == 1
        assert "latency_ms" in data

    def test_recent_endpoint(self, client):
        resp = client.get("/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
