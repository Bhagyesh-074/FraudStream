# FraudStream — Changelog

All entries are indexed sequentially and never reordered.

---

## [0001] — 2026-09-21 — Phase 1: Repo scaffolding

- Created directory structure: `src/eda/`, `tests/`, `data/`, `notebooks/`
- Added `.gitignore` (data/ excluded), `requirements.txt` (Phase 1 deps only)
- Package init files for `src`, `src.eda`, `tests`

## [0002] — 2026-09-21 — Phase 1: Dataset download helper

- `src/eda/download_data.py`: Checks dataset existence at `data/creditcard.csv`
- Prints manual download instructions if missing
- No auto-download — dataset is manually placed

## [0003] — 2026-09-21 — Phase 1: Statistical profiling

- `src/eda/profiling.py`: Dataset loading, summary, statistical profile
  (mean/std/skew/kurtosis/percentiles), fraud-vs-legit comparison with
  effect sizes, amount outlier analysis with explicit signal-vs-noise reasoning
- Confirmed: 284,807 rows, 31 cols, 492 frauds, 0.1727% fraud rate, 0 nulls

## [0004] — 2026-09-21 — Phase 1: Cost matrix

- `src/eda/cost_matrix.py`: `CostMatrix` class calibrated from data
- FN cost = $122.21 (avg fraud amount), FP cost = $10.00 (customer friction)
- Cost ratio 12.2:1 — tolerate up to 12 false alarms per missed fraud
- `compute_expected_cost()` for business-cost evaluation of confusion matrices

## [0005] — 2026-09-21 — Phase 1: Imbalance diagnostics

- `src/eda/imbalance.py`: Precise imbalance ratio (577.9:1),
  naive baseline accuracy (99.83%), formatted diagnostic report
- Documents scale_pos_weight and resampling implications for Phase 2

## [0006] — 2026-09-21 — Phase 1: EDA runner

- `src/eda/run_eda.py`: Runs full analysis pipeline, prints all results,
  saves to `data/eda_results.json`

## [0007] — 2026-09-21 — Phase 1: Tests

- `tests/test_profiling.py`: 15 tests — dataset shape, columns, fraud count,
  summary keys, statistical profile structure, fraud-vs-legit comparison,
  amount outlier analysis
- `tests/test_cost_matrix.py`: 15 tests — defaults, calibration, cost
  computation, real-data validation
- `tests/test_imbalance.py`: 14 tests — synthetic ratio checks, real-data
  ratio/accuracy, diagnostic report content
- All 44 tests passing

## [0008] — 2026-09-21 — Phase 1: Documentation

- `README.md`: Project overview, manual dataset download instructions,
  quick start, architecture table, key findings
- `PROJECT_CONTEXT.md`: Cold-start context for new sessions
- `CHANGELOG.md`: This file
- `IMPLEMENTATION_WALKTHROUGH.md`: Textbook-style Phase 1 walkthrough

## [0009] — 2026-09-21 — Phase 2: Dependencies and requirements

- Updated `requirements.txt` with Phase 2 dependencies: `xgboost`, `tensorflow`,
  `mlflow`, `pandera`, `pydantic`.

## [0010] — 2026-09-21 — Phase 2: Data cleaning & feature engineering modules

- `src/pipeline/cleaning.py`: `validate_data()`, `log_transform_amount()` using
  `np.log1p` to handle skewness and zero amounts, `engineer_time_features()`
  for cyclical `hour_of_day`, and `prepare_features()`.
- `src/pipeline/features.py`: `build_feature_matrix()` producing 30 model features
  (V1-V28, log_amount, hour_of_day), `time_based_split()` for 80/20 temporal split,
  `compute_scale_pos_weight()` from train split, `split_report()` with explicit
  low fraud count warning flag (<80 threshold).
- Documented explicit deferral of rolling velocity features to Phase 3 streaming state.

## [0011] — 2026-09-21 — Phase 2: Model training & semi-supervised autoencoder

- `src/pipeline/train.py`: Cost-sensitive XGBoost baseline with dynamically calculated
  `scale_pos_weight = 545.4`.
- Keras deep autoencoder (30 -> 16 -> 8 -> 16 -> 30) trained strictly on
  227,428 Class=0 legitimate transactions from TRAIN split only (zero leakage).
- Reconstruction error computation: 74.9x separation on train set (0.265 vs 19.878),
  20.0x separation on test set (0.299 vs 5.983).
- Hybrid ensemble XGBoost model combining 30 base features + reconstruction error.
- Local MLflow experiment tracking (`fraudstream-phase2`).

## [0012] — 2026-09-21 — Phase 2: Cost-based evaluation & slice diagnostics

- `src/pipeline/evaluate.py`: `compute_metrics()` (AUC-PR, AUC-ROC),
  `find_cost_optimal_threshold()` against calibrated business cost matrix
  (FN=$122.21, FP=$10.00), and `slice_evaluation()` across Amount bands and
  hour of day windows.
- Optimal threshold identified at $t = 0.0310$ (saving $211.06 vs default 0.5
  on test set; 81.33% recall, 56.48% precision).

## [0013] — 2026-09-21 — Phase 2: Pipeline unit tests

- `tests/test_pipeline.py`: 25 unit tests covering validation checks, log amount
  transforms, cyclic time features, feature column consistency, chronological
  split integrity, non-hardcoded `scale_pos_weight`, and cost-optimal thresholding.
- Full test suite: 69 passing unit tests (44 Phase 1 + 25 Phase 2).

## [0014] — 2026-09-21 — Phase 2: Documentation & verification

- Updated `IMPLEMENTATION_WALKTHROUGH.md` with Section 2 textbook-style documentation.
- Updated `PROJECT_CONTEXT.md` with Phase 2 status and benchmark metrics.

## [0015] — 2026-09-21 — Phase 2: Paired bootstrap analysis & statistical equivalence

- Implemented `bootstrap_auc_pr_comparison()` and `format_bootstrap_report()` in `src/pipeline/evaluate.py`.
- Added 4 unit tests in `tests/test_pipeline.py` (total test suite: 73 passing unit tests).
- Ran 1,000 paired bootstrap resamples on the test set:
  - XGBoost 95% CI: [0.7084, 0.8813], Ensemble 95% CI: [0.6991, 0.8748].
  - Paired difference 95% CI: [-0.0148, +0.0350] (spans zero; P(Ens > XGB) = 32.8%).
- Confirmed models are statistically indistinguishable ($p > 0.05$).
- Selected XGBoost Baseline via Occam's razor / engineering parsimony (lower latency,
  zero deep learning runtime dependencies in streaming inference).
- Saved raw predictions to `data/test_predictions.csv` and bootstrap iterations to `data/bootstrap_auc_pr_results.csv`.
- Updated `IMPLEMENTATION_WALKTHROUGH.md` and `PROJECT_CONTEXT.md`.

## [0016] — 2026-09-21 — Phase 3: Kafka producer & immutable raw Parquet zone consumer

- `docker-compose.yml`: Deployed Apache Kafka 3.7.0 in KRaft mode (single node, no ZooKeeper dependency) listening on `localhost:9092`.
- `requirements.txt`: Added `confluent-kafka>=2.0` and `pyarrow>=12.0`.
- `src/streaming/producer.py`: Implemented chronological replay producer publishing 31-column JSON transaction events to topic `transactions-raw` with configurable inter-message throttling and non-blocking delivery callbacks.
- `src/streaming/raw_consumer.py`: Implemented plain Python raw zone consumer micro-batching events into date/hour-partitioned Parquet files (`data/raw_zone/date=YYYY-MM-DD/hour=HH/batch_<ts>_<uuid>.parquet`).
- Immutability guarantee: Verified zero transformations applied in consumer; preserved historical `Time` column unmodified.
- `tests/test_streaming.py`: Added 8 unit and integration tests (full test suite: 81 passing unit/integration tests).
- Verified live dual streaming: Produced and consumed 500 records concurrently at 382 msg/s, writing 2 Parquet batches with 100% attribute fidelity against source CSV.
- Updated `IMPLEMENTATION_WALKTHROUGH.md` with Section 3 and `PROJECT_CONTEXT.md`.

## [0017] — 2026-09-21 — Phase 4: Feature store (versioned Parquet) & train/serve consistency refactor

- `src/features/feature_logic.py`: Extracted centralized `compute_features()` supporting `dict` (real-time scoring), `pd.Series`, and `pd.DataFrame` (batch training). Enforced strict label independence (no `Class` requirement), eliminating train/serve skew at the architectural level.
- Refactored `src/pipeline/cleaning.py` and `src/pipeline/features.py` to delegate 100% of feature extraction to `feature_logic.py`, removing duplicate feature logic.
- `src/features/feature_store.py`: Built lightweight versioned Parquet feature store with JSON manifest tracking (`data/feature_store/features_v{N}.parquet` and `manifest.json`). Registered Version 1 baseline (284,807 rows x 30 features).
- `tests/test_features.py`: Added 12 unit tests verifying input normalization, error handling, manifest versioning, unlabeled payload compatibility, single-record vs. batch numerical parity, and training regression invariance.
- Full test suite: 93 passing tests (`pytest tests/ -v`).
- Numerical consistency proof: Demonstrated 100% bit-for-bit parity (`0.00e+00` absolute difference across all 30 features) between single-record and batch evaluation on real raw transactions.
- Model invariance verified: Re-trained XGBoost test AUC-PR equals Phase 2 baseline (`0.799586` within machine precision $2 \times 10^{-7}$).
- Updated `IMPLEMENTATION_WALKTHROUGH.md` with Section 4 and `PROJECT_CONTEXT.md`.

## [0018] — 2026-09-22 — Phase 5: Real-time FastAPI + Kafka consumer scoring service

- `src/serving/model_loader.py`: File-based model and threshold loading with fail-fast behavior (no silent defaults). Loads XGBoost JSON model and `threshold_config.json` produced by training pipeline.
- `src/serving/scoring_consumer.py`: Kafka scoring consumer subscribing to `transactions-raw` (separate `fraudstream-scoring-group`), applying `compute_features()` from Phase 4, scoring via XGBoost `predict_proba`, and publishing scored payloads to `transactions-scored` topic. Thread-safe stats accumulator with latency percentiles and ring buffer.
- `src/serving/api.py`: FastAPI observability layer with `/health`, `/stats`, and `/recent` endpoints. Consumer runs in background thread; no synchronous `/predict` endpoint.
- `src/pipeline/train.py`: Added model serialization — saves XGBoost model to `models/xgboost_baseline.json` and threshold config to `models/threshold_config.json` at end of training pipeline.
- `src/streaming/producer.py`: Added `--start-row` CLI option for targeted row-range replay (fraud verification).
- `tests/test_serving.py`: 19 unit tests covering model loading, scoring logic, payload schema, latency SLA, stats accumulator, thread safety, test-split fraud records, and FastAPI endpoints.
- Full test suite: 112 passing tests (`pytest tests/ -v`).
- Honest side-by-side fraud verification (train-set seen vs. test-set unseen):
  - In-sample (Train split, seen data, Time <= 145,247s): 417/417 correctly flagged (100.0% recall, representing in-sample fit / near-memorization).
  - Out-of-sample (Held-out Test split, unseen data, Time > 145,247s): **61/75 correctly flagged (81.33% recall)**, exactly matching Phase 2's offline test benchmark (61/75, 81.33%) down to the exact individual transaction, mathematically proving zero train/serve skew in live scoring.
  - Test set scoring latency: p50=7.17ms, p95=8.21ms (well within 15ms SLA).
- Updated `IMPLEMENTATION_WALKTHROUGH.md` with Section 5 and `PROJECT_CONTEXT.md`.

## [0019] — 2026-09-22 — Phase 6: DVC versioning + MLflow Model Registry promotion gating
 
- `dvc`: Initialized DVC repository and tracked `data/creditcard.csv`, `data/feature_store/`, and `models/xgboost_baseline.json` via DVC pointers (`.dvc` files). Configured local remote storage (`c:/Projects/FraudStream/dvc_remote`), verifying `dvc status`, `dvc push`, and `dvc pull`.
- `.gitignore`: Updated to allow Git tracking of `.dvc` pointer files while ignoring heavy binary data files, local DVC remotes, and `mlflow.db`.
- `src/pipeline/train.py`: Updated tracking URI to embedded SQLite backend (`sqlite:///c:/Projects/FraudStream/mlflow.db`) supporting MLflow Model Registry. Implemented `register_and_gate_model()` and `evaluate_and_gate_candidate()` with strict promotion gating (promoting candidate to `Production` only if test AUC-PR is strictly superior to active production model; archiving previous production models upon promotion).
- `src/serving/model_loader.py`: Updated `load_model()` to query `models:/fraudstream-xgboost/Production` from MLflow Model Registry with explicit warning log and `local_fallback` if registry is unavailable or custom directory is specified. Added `get_active_model_source()`.
- `src/serving/api.py`: Updated `/health` endpoint to expose `"model_source": "registry" | "local_fallback"`, guaranteeing zero silent fallback.
- Gating demonstration: Verified initial cold-start promotion (v1 -> Production), rejection of synthetic inferior candidate v2 (AUC-PR 0.750000 -> Staging), and promotion of synthetic superior candidate v3 (AUC-PR 0.812500 -> Production with v1 -> Archived), followed by manual administrative re-alignment (direct MLflow transition call outside the gate) back to genuine baseline v1.
- Lineage audit demonstration: Executed 4-tier reproducibility trace for test fraud transaction at row 229712 across Git commit SHA (`746d785`), DVC data MD5 (`e90efcb8...`), Feature Store version (`v1`), and MLflow Model Registry (`fraudstream-xgboost` v1, Production, Run `e57d83b8...`).
- `tests/test_versioning.py`: Added 12 unit tests verifying DVC pointer integrity, `dvc status`, promotion gating rules, history retention in archiving, and serving loader governance.
- Full test suite: 124 passing tests (`pytest tests/ -v`).
- Updated `IMPLEMENTATION_WALKTHROUGH.md` with Section 6 and `PROJECT_CONTEXT.md`.
