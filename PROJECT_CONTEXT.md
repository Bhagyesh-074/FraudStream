# FraudStream — Project Context

_This document is the cold-start context file. A new model session should
read this first to understand the project's purpose, architecture, design
reasoning, and current phase status._

## Purpose

FraudStream is an industry-style end-to-end fraud detection pipeline, built
as a portfolio project for a Data Scientist role application at v4c.ai.

The v4c.ai guide for this role emphasizes:
- **pandas/scikit-learn/NumPy fluency** — core data science literacy
- **Business-driven cost-based evaluation** — not just model accuracy, but
  real-world cost of errors
- **Statistical rigor in data cleaning** — understand the data before
  touching it, document reasoning for cleaning decisions
- **Communication clarity** — explain findings in plain language, not just
  code output

These emphasis points drove every architecture and tooling choice below.

## Architecture

The architecture is deliberately lighter-weight than a typical "big data"
pipeline. The goal is to demonstrate DS competency, not infrastructure
engineering.

| Component | Tool | Reasoning |
|-----------|------|-----------|
| Ingestion | Kafka (Docker Compose) | Industry-standard event streaming; Docker keeps setup simple |
| Consumption | Plain Python consumer | NOT Spark — demonstrates pandas fluency, keeps complexity manageable |
| Validation | Pandera | Schema validation at ingestion boundary; lightweight, pandas-native |
| Modeling | XGBoost + Keras autoencoder | Gradient boosting for tabular data (industry workhorse) + autoencoder for anomaly detection (complementary approach) |
| Experiment tracking | MLflow | Standard experiment tracking; records hyperparameters, metrics, artifacts |
| Data versioning | DVC | Versioned dataset snapshots tied to git commits |
| Serving | FastAPI | Lightweight async API; easy to deploy, easy to test |
| Orchestration | Airflow | DAG-based workflow management; industry standard for batch pipelines |
| Monitoring | Evidently | Data/model drift detection; triggers retraining when distributions shift |

## Dataset

Kaggle Credit Card Fraud Detection dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- 284,807 transactions, 31 columns
- Features: Time, V1-V28 (PCA-transformed), Amount, Class (0=legit, 1=fraud)
- Fraud rate: 0.1727% (492 frauds)
- No null values, all numeric

## Phase Plan

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | EDA, statistical profiling, cost matrix, imbalance diagnostics | **COMPLETE** |
| **Phase 2** | Batch pipeline — cleaning, feature engineering, training, evaluation | **COMPLETE** |
| **Phase 3** | Kafka producer + raw Parquet zone consumer | **COMPLETE** |
| **Phase 4** | Feature store (versioned Parquet) + train/serve consistency | **COMPLETE** |
| **Phase 5** | Real-time FastAPI + Kafka consumer scoring service | **COMPLETE** |
| **Phase 6** | DVC + MLflow Model Registry promotion gating | **COMPLETE** |
| Phase 7 | Airflow orchestration + Evidently monitoring, drift-triggered retrain | NOT STARTED |

## Phase 1 Key Outputs

1. **Statistical profiling**: Distribution analysis of all 30 features,
   fraud vs legitimate comparison, effect size ranking
2. **Cost matrix**: FN=$122.21 (avg fraud amount), FP=$10.00 (customer
   friction), ratio 12.2:1
3. **Imbalance diagnostics**: 577.9:1 ratio, naive baseline 99.83% accuracy
4. **Amount outlier reasoning**: Outliers show HIGHER fraud rate (0.29% vs
   0.16%) — signal, not noise; Phase 2 should scale/clip, not drop
5. **Top discriminating features**: V17 (effect size -8.9), V14 (-7.8),
   V12 (-6.6), V10 (-5.4), V16 (-4.9)

## Phase 2 Key Outputs

1. **Time-based split**: Chronological 80/20 partition (Train: 227,845 with 417
   frauds; Test: 56,962 with 75 frauds). Explicitly flagged test fraud count <80.
2. **Feature transforms**: `log_amount` ($\ln(1 + \text{Amount})$) and `hour_of_day`
   cyclic feature. Velocity features deferred to Phase 3 streaming state.
3. **Imbalance-aware XGBoost**: `scale_pos_weight = 545.4` dynamically computed from
   training split.
4. **Zero-leakage Autoencoder**: Trained strictly on 227,428 legitimate transactions
   from the training split. Achieved 74.9x reconstruction error separation on train
   and 20.0x separation on test.
5. **Model comparison & bootstrap analysis**: Point estimates: XGBoost AUC-PR = 0.7996,
   Ensemble AUC-PR = 0.7931. 1,000 paired bootstrap resamples proved models are
   statistically indistinguishable (95% CI of diff: [-0.0148, +0.0350]; P(Ens>XGB) = 32.8%).
   XGBoost Baseline selected via Occam's razor (lower latency, no neural net dependencies).
6. **Cost-optimal threshold**: Optimal $t = 0.0310$ (vs default 0.50), cutting business
   cost from $2,392.02 to $2,180.96 ($211.06 savings on test set; 81.33% recall, 56.48% precision).
7. **Slice diagnostics**: Evaluated across 5 Amount tiers and 2 time windows.
8. **Testing**: 73 passing unit tests (44 Phase 1 + 29 Phase 2).

## Phase 3 Key Outputs

1. **KRaft Mode Kafka**: Deployed lightweight Apache Kafka 3.7.0 without ZooKeeper via Docker Compose (`docker-compose.yml`, port 9092).
2. **Chronological Replay Producer**: `src/streaming/producer.py` streams 31-column JSON transaction events preserving exact `Time` ordering at >380 msg/s.
3. **Immutable Raw Parquet Consumer**: `src/streaming/raw_consumer.py` micro-batches events into `data/raw_zone/date=YYYY-MM-DD/hour=HH/` Parquet files using PyArrow with zero transformation.
4. **Time & Value Fidelity**: Historical `Time` seconds elapsed preserved 100% unmodified; ingestion wall-clock time used strictly for directory partitioning.
5. **Testing**: 81 passing unit and integration tests (`pytest tests/ -v`).

## Phase 4 Key Outputs

1. **Shared Feature Module**: `src/features/feature_logic.py:compute_features` serves both batch DataFrames and single-record dicts/Series with zero duplicated logic.
2. **Label Independence**: Supports unlabeled payloads without `Class` key, eliminating runtime errors during real-time scoring.
3. **Versioned Parquet Feature Store**: `src/features/feature_store.py` manages immutable snapshots (`data/feature_store/features_v1.parquet`) and manifest metadata (`manifest.json`).
4. **Zero Train/Serve Skew Proof**: Exact side-by-side numerical comparison verified $0.00 \times 10^0$ maximum difference across all 30 features between single-record live scoring and batch training.
5. **Model Invariance**: Re-trained XGBoost on refactored features reproduced Phase 2 baseline AUC-PR = `0.799586` (difference $< 2 \times 10^{-7}$).
6. **Testing**: 93 passing unit and integration tests (`pytest tests/ -v`).
## Phase 5 Key Outputs

1. **Model Serialization**: `src/pipeline/train.py` exports XGBoost model to `models/xgboost_baseline.json` and cost-optimal threshold to `models/threshold_config.json` with fail-fast loading in `src/serving/model_loader.py`.
2. **Real-time Kafka Scoring Consumer**: `src/serving/scoring_consumer.py` reads from `transactions-raw` (group `fraudstream-scoring-group`), computes features on unlabeled records via `compute_features()`, scores at sub-10ms latency, and emits enriched results to `transactions-scored`.
3. **Thread-Safe Metrics & Telemetry**: Accumulated stats (`records_scored`, `frauds_flagged`, `latencies`) with ring buffer of recent transactions, thread-safe via `threading.Lock`.
4. **FastAPI Observability**: `src/serving/api.py` exposes `/health` (model, threshold, uptime), `/stats` (throughput, fraud rate, p50/p95 latency), and `/recent` (last N scored events) with background consumer lifecycle.
5. **Honest Fraud Verification (Train vs Test)**: Tested scoring consumer across splits:
   - In-sample (Train split, seen, Time <= 145,247s): 417/417 (100.0%) correctly flagged (near-memorization).
   - Out-of-sample (Test split, genuinely unseen, Time > 145,247s): **61/75 (81.33%) correctly flagged**, confirming exact parity with Phase 2 offline test set recall (81.33% = 61/75) with zero train/serve skew; p50 scoring latency ~7.17ms.

## Phase 6 Key Outputs

1. **DVC Versioning & Tracking**: Initialized DVC repository tracking `data/creditcard.csv`, `data/feature_store/`, and `models/xgboost_baseline.json` via pointer `.dvc` files pushed to local remote (`dvc_remote`). Verified `dvc status` clean and push/pull operational.
2. **Model Registry & Promotion Gating**: Replaced file store with embedded SQLite tracking backend (`sqlite:///mlflow.db`) supporting MLflow Model Registry (`fraudstream-xgboost`). Implemented `evaluate_and_gate_candidate()`: candidate promoted to `Production` only if test AUC-PR strictly beats current production model; previous production model archived.
3. **Governance in Serving Loader**: Updated `src/serving/model_loader.py` to query `models:/fraudstream-xgboost/Production` from the registry with non-silent warning log on fallback. Exposes active source via `/health` endpoint (`"model_source": "registry" | "local_fallback"`).
4. **4-Tier Reproducibility Lineage Audit**: Demonstrated complete, deterministic end-to-end trace for row 229712 across Git commit SHA (`746d785`), DVC dataset MD5 (`e90efcb8...`), Feature Store version snapshot (`v1`), and MLflow Model Registry (`fraudstream-xgboost` v1, Production, optimal threshold $t=0.0310$).
5. **Testing**: 124 passing tests across full regression suite (`pytest tests/ -v`).


## Conventions

- **CHANGELOG.md**: Indexed entries [0001], [0002]... never reordered
- **IMPLEMENTATION_WALKTHROUGH.md**: Textbook-style explanatory document,
  one section per phase, never edited retroactively
- **Tests**: pytest in tests/, skip gracefully if dataset is absent
- **Data**: gitignored, manual download from Kaggle
