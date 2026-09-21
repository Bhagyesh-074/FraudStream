"""
Model training module for the FraudStream batch pipeline.

Trains two models:
1. XGBoost classifier with scale_pos_weight from Phase 1's imbalance ratio
2. Keras autoencoder trained ONLY on legitimate (Class=0) transactions from
   the TRAINING split — reconstruction error is used as an anomaly score

The autoencoder's reconstruction error is fed as an additional feature into
a second XGBoost model (the "ensemble" model), combining supervised and
unsupervised signals.

All runs are logged to MLflow (local tracking directory).

CRITICAL: The autoencoder is trained on Class=0 rows from the TRAIN split
ONLY — no test data is used in any training step. This is verified in tests.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from src.eda.profiling import load_dataset
from src.eda.cost_matrix import CostMatrix
from src.pipeline.cleaning import prepare_features, get_feature_columns
from src.pipeline.features import (
    build_feature_matrix,
    time_based_split,
    compute_scale_pos_weight,
    split_report,
)
from src.pipeline.evaluate import (
    compute_metrics,
    find_cost_optimal_threshold,
    slice_evaluation,
    format_evaluation_report,
    bootstrap_auc_pr_comparison,
    format_bootstrap_report,
)

# Suppress TF info/warning noise and MLflow file store warning
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _get_mlflow_uri() -> str:
    """Return the MLflow tracking URI (local directory)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return f"file:///{project_root / 'mlruns'}"


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    experiment_name: str = "fraudstream-phase2",
    run_name: str = "xgboost-baseline",
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with imbalance-aware weighting.

    Args:
        X_train: Training features.
        y_train: Training labels.
        scale_pos_weight: Weight for positive (fraud) class.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        Trained XGBClassifier.
    """
    mlflow.set_tracking_uri(_get_mlflow_uri())
    mlflow.set_experiment(experiment_name)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_train_fraud", int((y_train == 1).sum()))
        mlflow.log_param("n_train_legit", int((y_train == 0).sum()))
        mlflow.log_param("n_features", X_train.shape[1])

        model.fit(X_train, y_train)
        mlflow.xgboost.log_model(model, "model")

    return model


def train_autoencoder(
    X_train_legit: pd.DataFrame,
    encoding_dim: int = 14,
    epochs: int = 50,
    batch_size: int = 256,
    experiment_name: str = "fraudstream-phase2",
    run_name: str = "autoencoder",
):
    """Train a Keras autoencoder on legitimate-only training data.

    CRITICAL: This function receives ONLY legitimate (Class=0) rows from
    the TRAINING split. NO test data, NO fraud data. The autoencoder
    learns what "normal" transactions look like; at inference, fraud
    transactions should have higher reconstruction error (anomaly score).

    Architecture:
      Input(30) -> Dense(encoding_dim, relu) -> Dense(30, sigmoid)

    Args:
        X_train_legit: Feature matrix of legitimate training transactions ONLY.
        encoding_dim: Size of the bottleneck layer.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        Trained Keras model.
    """
    # Lazy import to avoid TF startup cost when not needed
    import tensorflow as tf
    from tensorflow import keras

    # Set seed for reproducible weight initialization and training
    keras.utils.set_random_seed(42)

    mlflow.set_tracking_uri(_get_mlflow_uri())
    mlflow.set_experiment(experiment_name)

    input_dim = X_train_legit.shape[1]
    X_np = X_train_legit.values.astype(np.float32)

    # Build autoencoder
    encoder_input = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(encoding_dim, activation="relu")(encoder_input)
    decoded = keras.layers.Dense(input_dim, activation="linear")(encoded)
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "architecture": f"{input_dim}-{encoding_dim}-{input_dim}",
            "encoding_dim": encoding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adam",
            "loss": "mse",
            "n_train_legit_samples": len(X_train_legit),
            "trained_on": "Class=0 from TRAIN split ONLY",
        })

        history = autoencoder.fit(
            X_np, X_np,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=0,
        )

        final_loss = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        mlflow.log_metric("final_train_loss", final_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)

    return autoencoder


def compute_reconstruction_error(
    autoencoder,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute per-sample reconstruction error (MSE) as anomaly score.

    Fraud transactions — which the autoencoder never saw during training —
    should have higher reconstruction error because they deviate from the
    "normal" patterns the autoencoder learned.

    Args:
        autoencoder: Trained Keras autoencoder.
        X: Feature matrix (any split — train or test).

    Returns:
        1D array of reconstruction errors (one per sample).
    """
    X_np = X.values.astype(np.float32)
    X_reconstructed = autoencoder.predict(X_np, verbose=0)
    mse = np.mean((X_np - X_reconstructed) ** 2, axis=1)
    return mse


def train_ensemble_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    recon_error_train: np.ndarray,
    scale_pos_weight: float,
    experiment_name: str = "fraudstream-phase2",
    run_name: str = "xgboost-ensemble",
) -> xgb.XGBClassifier:
    """Train XGBoost with reconstruction error as an additional feature.

    This combines the supervised signal (XGBoost on tabular features) with
    the unsupervised signal (autoencoder anomaly score) into a single model.

    Args:
        X_train: Original training features.
        y_train: Training labels.
        recon_error_train: Autoencoder reconstruction error per sample.
        scale_pos_weight: Weight for positive class.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        Trained XGBClassifier with n_features + 1 inputs.
    """
    mlflow.set_tracking_uri(_get_mlflow_uri())
    mlflow.set_experiment(experiment_name)

    X_augmented = X_train.copy()
    X_augmented["recon_error"] = recon_error_train

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_augmented.shape[1])
        mlflow.log_param("includes_recon_error", True)
        mlflow.log_param("n_train_samples", len(X_train))

        model.fit(X_augmented, y_train)
        mlflow.xgboost.log_model(model, "model")

    return model


def run_training_pipeline():
    """Orchestrate the full Phase 2 training and evaluation pipeline.

    Steps:
    1. Load data and apply cleaning/features
    2. Time-based train/test split
    3. Report split statistics (with exact fraud counts)
    4. Train XGBoost baseline
    5. Train autoencoder on TRAIN Class=0 ONLY
    6. Compute reconstruction errors
    7. Train ensemble XGBoost (with recon error feature)
    8. Evaluate all models
    9. Cost-based threshold selection
    10. Slice-based evaluation
    """
    print("=" * 70)
    print("  FRAUDSTREAM PHASE 2: TRAINING PIPELINE")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/10] Loading dataset...")
    df = load_dataset()
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ── 2. Time-based split ──────────────────────────────────────────
    print("\n[2/10] Time-based train/test split...")
    train_df, test_df = time_based_split(df, train_fraction=0.80)
    report = split_report(train_df, test_df)
    print(report)

    # ── 3. Build feature matrices ────────────────────────────────────
    print("\n[3/10] Building feature matrices...")
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    print(f"  Train features: {X_train.shape}")
    print(f"  Test features:  {X_test.shape}")

    # Prepare test_df with engineered features for slice evaluation
    test_df_prepared = prepare_features(test_df)

    # Compute scale_pos_weight from TRAINING set
    spw = compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight (from training set): {spw:.1f}")

    # ── 4. Train XGBoost baseline ────────────────────────────────────
    print("\n[4/10] Training XGBoost baseline...")
    xgb_model = train_xgboost(X_train, y_train, spw)
    xgb_prob_test = xgb_model.predict_proba(X_test)[:, 1]
    print(f"  XGBoost trained. Test predictions computed.")

    # ── 5. Train autoencoder ─────────────────────────────────────────
    print("\n[5/10] Training autoencoder on TRAIN Class=0 ONLY...")
    # CRITICAL: Only legitimate transactions from TRAINING split
    train_legit_mask = y_train == 0
    X_train_legit = X_train[train_legit_mask]
    n_train_legit = len(X_train_legit)
    n_train_fraud = int((y_train == 1).sum())
    print(f"  Autoencoder training data: {n_train_legit:,} legitimate transactions")
    print(f"  EXCLUDED from autoencoder training: {n_train_fraud} fraud + ALL test data")
    print(f"  CONFIRMATION: Autoencoder sees ONLY Class=0 from TRAIN split. No test leakage.")

    autoencoder = train_autoencoder(X_train_legit)
    print(f"  Autoencoder trained.")

    # ── 6. Compute reconstruction errors ─────────────────────────────
    print("\n[6/10] Computing reconstruction errors...")
    recon_error_train = compute_reconstruction_error(autoencoder, X_train)
    recon_error_test = compute_reconstruction_error(autoencoder, X_test)

    # Report reconstruction error stats by class
    train_fraud_mask = y_train == 1
    re_legit_mean = float(recon_error_train[train_legit_mask].mean())
    re_fraud_mean = float(recon_error_train[train_fraud_mask].mean()) if train_fraud_mask.sum() > 0 else 0.0
    print(f"  Recon error (train legit): mean={re_legit_mean:.6f}")
    print(f"  Recon error (train fraud): mean={re_fraud_mean:.6f}")
    print(f"  Ratio (fraud/legit):       {re_fraud_mean/re_legit_mean:.2f}x" if re_legit_mean > 0 else "")

    test_legit_mask_test = y_test == 0
    test_fraud_mask_test = y_test == 1
    re_test_legit = float(recon_error_test[test_legit_mask_test].mean())
    re_test_fraud = float(recon_error_test[test_fraud_mask_test].mean()) if test_fraud_mask_test.sum() > 0 else 0.0
    print(f"  Recon error (test legit):  mean={re_test_legit:.6f}")
    print(f"  Recon error (test fraud):  mean={re_test_fraud:.6f}")

    # ── 7. Train ensemble XGBoost ────────────────────────────────────
    print("\n[7/10] Training ensemble XGBoost (features + recon error)...")
    ensemble_model = train_ensemble_xgboost(
        X_train, y_train, recon_error_train, spw
    )
    # For ensemble test predictions, augment test features
    X_test_augmented = X_test.copy()
    X_test_augmented["recon_error"] = recon_error_test
    ensemble_prob_test = ensemble_model.predict_proba(X_test_augmented)[:, 1]
    print(f"  Ensemble XGBoost trained. Test predictions computed.")

    # ── 8. Evaluate both models ──────────────────────────────────────
    print("\n[8/10] Computing evaluation metrics...")

    # Calibrate cost matrix from full dataset (Phase 1 approach)
    cost_matrix = CostMatrix()
    cost_matrix.calibrate_from_data(df)
    print(f"  Cost matrix: FN=${cost_matrix.fn_cost:.2f}, FP=${cost_matrix.fp_cost:.2f}, "
          f"ratio={cost_matrix.cost_ratio:.1f}:1")

    # XGBoost baseline metrics
    xgb_metrics = compute_metrics(y_test, xgb_prob_test)
    print(f"\n  XGBoost Baseline:")
    print(f"    AUC-PR:  {xgb_metrics['auc_pr']:.6f}")
    print(f"    AUC-ROC: {xgb_metrics['auc_roc']:.6f}")

    # Ensemble metrics
    ens_metrics = compute_metrics(y_test, ensemble_prob_test)
    print(f"\n  Ensemble (XGBoost + recon error):")
    print(f"    AUC-PR:  {ens_metrics['auc_pr']:.6f}")
    print(f"    AUC-ROC: {ens_metrics['auc_roc']:.6f}")

    # Save test predictions for auditing
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "xgboost_prob": xgb_prob_test,
        "ensemble_prob": ensemble_prob_test,
    })
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    pred_path = data_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Test predictions saved to: {pred_path}")

    # ── 8b. Bootstrap Model Comparison (1,000 paired resamples) ─────
    print("\n[8b/10] Running 1,000 paired bootstrap iterations...")
    bootstrap_res = bootstrap_auc_pr_comparison(
        y_test.values,
        xgb_prob_test,
        ensemble_prob_test,
        name_a="XGBoost Baseline",
        name_b="Ensemble",
        n_bootstraps=1000,
        random_state=42,
    )

    # Save raw bootstrap iterations to CSV
    boot_df = pd.DataFrame({
        "iteration": np.arange(1, 1001),
        "xgboost_auc_pr": bootstrap_res["raw_scores_a"],
        "ensemble_auc_pr": bootstrap_res["raw_scores_b"],
        "diff_auc_pr": bootstrap_res["raw_diffs"],
    })
    boot_csv_path = data_dir / "bootstrap_auc_pr_results.csv"
    boot_df.to_csv(boot_csv_path, index=False)
    print(f"  Raw bootstrap samples saved to: {boot_csv_path}")

    # Print bootstrap comparison report
    print("\n" + format_bootstrap_report(bootstrap_res))

    # Print deciles for exact raw statistical inspection
    print("BOOTSTRAP PERCENTILES & DECILES:")
    print(f"{'Percentile':<12} {'XGBoost AUC-PR':<18} {'Ensemble AUC-PR':<18} {'Diff (XGB - Ens)':<18}")
    print("-" * 68)
    percentiles = [2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97.5]
    for p in percentiles:
        val_a = float(np.percentile(bootstrap_res["raw_scores_a"], p))
        val_b = float(np.percentile(bootstrap_res["raw_scores_b"], p))
        val_d = float(np.percentile(bootstrap_res["raw_diffs"], p))
        print(f"p{p:<10.1f} {val_a:<18.6f} {val_b:<18.6f} {val_d:<+18.6f}")

    # Print first 10 raw sample iterations as evidence
    print("\nSAMPLE RAW BOOTSTRAP ITERATIONS (first 10 of 1,000):")
    print(f"{'Iter':<6} {'XGBoost':<14} {'Ensemble':<14} {'Diff (XGB - Ens)':<16}")
    print("-" * 52)
    for i in range(10):
        print(f"#{i+1:<5} {bootstrap_res['raw_scores_a'][i]:<14.6f} {bootstrap_res['raw_scores_b'][i]:<14.6f} {bootstrap_res['raw_diffs'][i]:<+16.6f}")

    # Model selection reasoning based on evidence
    if bootstrap_res["statistically_significant"]:
        if bootstrap_res["diff"]["ci_lower"] > 0:
            best_name = "XGBoost Baseline"
            best_prob = xgb_prob_test
            best_metrics = xgb_metrics
            selection_reason = "XGBoost is statistically significantly superior (95% CI > 0)."
        else:
            best_name = "Ensemble"
            best_prob = ensemble_prob_test
            best_metrics = ens_metrics
            selection_reason = "Ensemble is statistically significantly superior (95% CI < 0)."
    else:
        # Statistically indistinguishable: invoke Occam's razor / engineering parsimony
        best_name = "XGBoost Baseline"
        best_prob = xgb_prob_test
        best_metrics = xgb_metrics
        selection_reason = (
            "Models are statistically indistinguishable (95% CI of difference spans 0). "
            "Selected XGBoost Baseline via Occam's razor: simpler operational architecture, "
            "lower inference latency, and avoids deep learning serving dependencies."
        )

    print(f"\n  Final Model Selected: {best_name}")
    print(f"  Selection Rationale:  {selection_reason}")

    # ── 9. Cost-based threshold selection ────────────────────────────
    print("\n[9/10] Finding cost-optimal threshold...")
    threshold_result = find_cost_optimal_threshold(
        y_test, best_prob, cost_matrix
    )

    # ── 10. Slice evaluation ─────────────────────────────────────────
    print("\n[10/10] Slice-based evaluation...")
    slices = slice_evaluation(
        test_df_prepared, best_prob,
        threshold_result["optimal_threshold"], cost_matrix
    )

    # ── Full report ──────────────────────────────────────────────────
    candidate_thresholds = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    full_report = format_evaluation_report(
        best_metrics, threshold_result, slices, cost_matrix,
        candidate_thresholds, y_test.values, best_prob,
    )
    print()
    print(full_report)

    # Log best model metrics to MLflow
    mlflow.set_tracking_uri(_get_mlflow_uri())
    mlflow.set_experiment("fraudstream-phase2")
    with mlflow.start_run(run_name=f"evaluation-{best_name.lower().replace(' ', '-')}"):
        mlflow.log_metrics({
            "auc_pr": best_metrics["auc_pr"],
            "auc_roc": best_metrics["auc_roc"],
            "optimal_threshold": threshold_result["optimal_threshold"],
            "min_business_cost": threshold_result["min_cost"],
            "recall_at_optimal": threshold_result["recall"],
            "precision_at_optimal": threshold_result["precision"],
        })
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("cost_fn", cost_matrix.fn_cost)
        mlflow.log_param("cost_fp", cost_matrix.fp_cost)

    print(f"\nMLflow tracking URI: {_get_mlflow_uri()}")
    print("Pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()
