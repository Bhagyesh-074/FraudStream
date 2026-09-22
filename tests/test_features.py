"""Tests for shared feature logic, train/serve consistency, and versioned feature store."""

import numpy as np
import pandas as pd
import pytest

from src.eda.download_data import DEFAULT_CSV_PATH
from src.features.feature_logic import (
    FEATURE_COLUMNS,
    PCA_FEATURES,
    ENGINEERED_FEATURES,
    REQUIRED_RAW_COLUMNS,
    compute_features,
)
from src.features.feature_store import (
    write_new_version,
    get_current_features,
    get_version,
    list_versions,
    get_manifest,
)
from src.pipeline.features import time_based_split, compute_scale_pos_weight
from src.pipeline.train import train_xgboost
from src.pipeline.evaluate import compute_metrics


@pytest.fixture
def sample_raw_record():
    """Sample raw transaction record with all 31 columns including Class."""
    record = {f"V{i}": float(i * 0.1) for i in range(1, 29)}
    record["Time"] = 3600.0  # 1 hour in seconds
    record["Amount"] = 149.62
    record["Class"] = 0
    return record


@pytest.fixture
def sample_batch_df(sample_raw_record):
    """Batch DataFrame containing multiple transaction records."""
    records = []
    for i in range(20):
        rec = sample_raw_record.copy()
        rec["Time"] = float(i * 1800)  # half-hour increments
        rec["Amount"] = float(10.0 + i * 15.0)
        rec["V1"] = float(-1.0 + i * 0.05)
        rec["Class"] = 1 if i % 10 == 0 else 0
        records.append(rec)
    return pd.DataFrame(records)


class TestTrainServeConsistency:
    """Core verification: single-record feature computation matches batch computation."""

    def test_single_record_matches_batch_computation(self, sample_batch_df):
        """Single-record dict and full batch DataFrame produce identical feature values for the same row."""
        target_idx = 5
        target_record = sample_batch_df.iloc[target_idx].to_dict()

        feats_single = compute_features(target_record)
        feats_batch = compute_features(sample_batch_df)

        assert len(feats_single) == 1
        assert list(feats_single.columns) == FEATURE_COLUMNS
        assert list(feats_batch.columns) == FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            val_single = feats_single[col].iloc[0]
            val_batch = feats_batch[col].iloc[target_idx]
            assert abs(val_single - val_batch) < 1e-6, (
                f"Train/serve skew detected in feature '{col}': single={val_single} vs batch={val_batch}"
            )

    def test_real_data_single_vs_batch_consistency(self):
        """Verify train/serve consistency on real dataset records if available."""
        if not DEFAULT_CSV_PATH.is_file():
            pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

        df_full = pd.read_csv(DEFAULT_CSV_PATH, nrows=500)
        test_indices = [0, 50, 100, 250, 499]

        feats_batch = compute_features(df_full)

        for idx in test_indices:
            record_dict = df_full.iloc[idx].to_dict()
            feats_single = compute_features(record_dict)

            for col in FEATURE_COLUMNS:
                single_val = float(feats_single[col].iloc[0])
                batch_val = float(feats_batch[col].iloc[idx])
                assert abs(single_val - batch_val) < 1e-6, (
                    f"Mismatch on row {idx} for feature '{col}': single={single_val}, batch={batch_val}"
                )


class TestUnlabeledLiveScoringPayload:
    """Verify live scoring payloads without a 'Class' column succeed with identical feature output."""

    def test_unlabeled_payload_succeeds_without_class_key(self, sample_raw_record):
        """compute_features must process a dict without 'Class' key without KeyError."""
        unlabeled_record = {k: v for k, v in sample_raw_record.items() if k != "Class"}
        assert "Class" not in unlabeled_record

        feats_unlabeled = compute_features(unlabeled_record)
        assert len(feats_unlabeled) == 1
        assert list(feats_unlabeled.columns) == FEATURE_COLUMNS

    def test_unlabeled_produces_identical_features_to_labeled_record(self, sample_raw_record):
        """Unlabeled payload produces bit-for-bit identical features to labeled counterpart."""
        labeled_record = sample_raw_record.copy()
        unlabeled_record = {k: v for k, v in sample_raw_record.items() if k != "Class"}

        feats_labeled = compute_features(labeled_record)
        feats_unlabeled = compute_features(unlabeled_record)

        pd.testing.assert_frame_equal(feats_labeled, feats_unlabeled)


class TestInputNormalizationAndValidation:
    """Test handling of different input types and validation error guards."""

    def test_series_input_produces_same_output_as_dict(self, sample_raw_record):
        """pd.Series input produces identical features to dict input."""
        series_record = pd.Series(sample_raw_record)
        feats_dict = compute_features(sample_raw_record)
        feats_series = compute_features(series_record)
        pd.testing.assert_frame_equal(feats_dict, feats_series)

    def test_unsupported_type_raises_type_error(self):
        """Passing unsupported data type raises TypeError."""
        with pytest.raises(TypeError, match="Expected dict, Series, or DataFrame"):
            compute_features([1, 2, 3])

    def test_missing_required_column_raises_value_error(self, sample_raw_record):
        """Omitting a required feature column raises ValueError."""
        incomplete = sample_raw_record.copy()
        del incomplete["V14"]
        with pytest.raises(ValueError, match="Missing required raw columns"):
            compute_features(incomplete)

    def test_null_values_raise_value_error(self, sample_raw_record):
        """Input containing nulls in required columns raises ValueError."""
        corrupt = sample_raw_record.copy()
        corrupt["Amount"] = None
        with pytest.raises(ValueError, match="Found null values in raw columns"):
            compute_features(corrupt)


class TestFeatureStoreVersioning:
    """Test Parquet feature store versioning and manifest tracking."""

    def test_initial_write_creates_v1_and_manifest(self, tmp_path, sample_batch_df):
        """First write creates features_v1.parquet and initializes manifest."""
        features = compute_features(sample_batch_df)
        v = write_new_version(features, "Initial baseline features", store_dir=tmp_path)
        assert v == 1

        manifest = get_manifest(tmp_path)
        assert manifest["current_version"] == 1
        assert "1" in manifest["versions"]
        assert manifest["versions"]["1"]["filename"] == "features_v1.parquet"
        assert manifest["versions"]["1"]["n_rows"] == len(sample_batch_df)
        assert manifest["versions"]["1"]["n_columns"] == 30

        loaded = get_current_features(store_dir=tmp_path)
        pd.testing.assert_frame_equal(features, loaded)

    def test_new_version_does_not_destroy_old_version(self, tmp_path, sample_batch_df):
        """Writing version 2 increments version and preserves version 1 intact."""
        features_v1 = compute_features(sample_batch_df.iloc[:10])
        v1 = write_new_version(features_v1, "Version 1", store_dir=tmp_path)
        assert v1 == 1

        features_v2 = compute_features(sample_batch_df)
        v2 = write_new_version(features_v2, "Version 2 with more rows", store_dir=tmp_path)
        assert v2 == 2

        manifest = get_manifest(tmp_path)
        assert manifest["current_version"] == 2
        assert len(manifest["versions"]) == 2

        loaded_v1 = get_version(1, store_dir=tmp_path)
        loaded_v2 = get_version(2, store_dir=tmp_path)
        assert len(loaded_v1) == 10
        assert len(loaded_v2) == len(sample_batch_df)

        curr = get_current_features(store_dir=tmp_path)
        assert len(curr) == len(sample_batch_df)

    def test_list_versions_returns_all_metadata(self, tmp_path, sample_batch_df):
        """list_versions returns complete version history dictionary."""
        feats = compute_features(sample_batch_df)
        write_new_version(feats, "First commit", store_dir=tmp_path)
        write_new_version(feats, "Second commit", store_dir=tmp_path)

        v_dict = list_versions(store_dir=tmp_path)
        assert "1" in v_dict and "2" in v_dict
        assert v_dict["1"]["description"] == "First commit"
        assert v_dict["2"]["description"] == "Second commit"


class TestModelRegressionAgainstPhase2:
    """Verify that re-running training via refactored shared logic preserves Phase 2 AUC-PR."""

    def test_auc_pr_matches_phase2_baseline(self):
        """XGBoost test AUC-PR must equal 0.799586 within 0.0001 (zero metric drift)."""
        if not DEFAULT_CSV_PATH.is_file():
            pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

        df = pd.read_csv(DEFAULT_CSV_PATH)
        train_df, test_df = time_based_split(df, train_fraction=0.80)

        # Build feature matrices using the refactored shared compute_features function
        X_train = compute_features(train_df)
        y_train = train_df["Class"]
        X_test = compute_features(test_df)
        y_test = test_df["Class"]

        spw = compute_scale_pos_weight(y_train)
        model = train_xgboost(X_train, y_train, spw, experiment_name="fraudstream-phase4-regression")
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test.values, y_prob)
        recomputed_auc_pr = metrics["auc_pr"]

        PHASE2_BENCHMARK_AUC_PR = 0.799586
        assert abs(recomputed_auc_pr - PHASE2_BENCHMARK_AUC_PR) < 0.0001, (
            f"Regression detected: Recomputed AUC-PR {recomputed_auc_pr:.6f} differs from Phase 2 baseline {PHASE2_BENCHMARK_AUC_PR:.6f}"
        )
