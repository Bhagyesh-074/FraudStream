# FraudStream

**Industry-style end-to-end fraud detection pipeline** — built for a Data Scientist role application (v4c.ai).

Demonstrates pandas/scikit-learn/NumPy fluency, business-driven cost-based evaluation, statistical rigor in data cleaning, and communication clarity.

## Current Status: Phase 6 Complete (Ready for Phase 7)

Completed:
- **Phase 1**: Exploratory data analysis, statistical profiling, cost matrix calibration, class imbalance diagnostics.
- **Phase 2**: Chronological batch pipeline, feature engineering, XGBoost baseline vs Autoencoder ensemble, cost-optimal thresholding.
- **Phase 3**: Real-time Kafka streaming (KRaft mode), replay producer, raw Parquet lake consumer with time-based partitioning.
- **Phase 4**: Shared feature store with zero train/serve skew proof (bit-for-bit parity), versioned Parquet feature tables.
- **Phase 5**: Real-time scoring consumer on `transactions-raw` topic, FastAPI observability plane, honest side-by-side held-out test fraud verification.
- **Phase 6**: DVC data & feature versioning with content-addressable storage, MLflow Model Registry promotion gating, 4-tier lineage reproducibility audit.

## Quick Start

### 1. Prerequisites

- Python 3.12+
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

This project uses the **Kaggle Credit Card Fraud Detection** dataset.

**Manual download steps:**

1. Go to: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **"Download"** (requires a free Kaggle account — sign up at [kaggle.com](https://www.kaggle.com) if needed)
3. Extract the downloaded ZIP file
4. Place `creditcard.csv` in the `data/` directory:
   ```
   FraudStream/
     data/
       creditcard.csv    <-- place here
   ```

**Expected file details:**
- Filename: `creditcard.csv`
- Size: ~150 MB
- Shape: 284,807 rows × 31 columns
- Columns: `Time`, `V1`–`V28` (PCA-transformed features), `Amount`, `Class` (0=legitimate, 1=fraud)
- Fraud rate: 0.1727% (492 out of 284,807 transactions)

> **Note:** The `data/` directory is gitignored. The dataset is not included in the repository.

### 4. Run EDA

```bash
python -m src.eda.run_eda
```

This runs the full Phase 1 analysis pipeline:
- Dataset summary and validation
- Statistical profiling of all features
- Fraud vs legitimate distribution comparison
- Amount outlier analysis
- Cost matrix calibration
- Class imbalance diagnostics

Results are printed to stdout and saved to `data/eda_results.json`.

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Project Architecture (Full Plan)

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | EDA, statistical profiling, cost matrix, imbalance diagnostics | **Complete** |
| **Phase 2** | Batch pipeline — cleaning, feature engineering, training, evaluation | **Complete** |
| **Phase 3** | Kafka producer + raw Parquet zone consumer | **Complete** |
| **Phase 4** | Feature store (versioned Parquet) + train/serve consistency | **Complete** |
| **Phase 5** | Real-time FastAPI + Kafka consumer scoring service | **Complete** |
| **Phase 6** | DVC + MLflow Model Registry promotion gating | **Complete** |
| Phase 7 | Airflow orchestration + Evidently monitoring, drift-triggered retrain | Planned |

## Repository Structure

```
FraudStream/
  data/                           # Raw dataset (gitignored)
    creditcard.csv                # Kaggle Credit Card Fraud dataset
    eda_results.json              # Computed EDA statistics
  src/
    eda/
      download_data.py            # Dataset existence check + instructions
      profiling.py                # Statistical profiling functions
      cost_matrix.py              # Business cost matrix definition
      imbalance.py                # Class imbalance diagnostics
      run_eda.py                  # Full EDA runner script
  tests/
    test_profiling.py             # Dataset shape/integrity tests
    test_cost_matrix.py           # Cost matrix validation tests
    test_imbalance.py             # Imbalance diagnostic tests
  notebooks/                      # Exploratory work (optional)
  requirements.txt                # Project dependencies
  README.md                       # This file
  .gitignore
```

## Key Findings (Phase 1)

- **Fraud rate**: 0.1727% (492 frauds out of 284,807 transactions)
- **Imbalance ratio**: 577.9:1 (legitimate to fraudulent)
- **Naive baseline accuracy**: 99.83% — completely useless despite the number
- **Cost matrix**: FN=$122.21 (missed fraud), FP=$10.00 (customer friction), ratio 12.2:1
- **Top discriminating features**: V17, V14, V12, V10, V16 (by effect size)
- **Amount outliers**: Higher fraud rate among outliers (0.29% vs 0.16%) — signal, not noise
