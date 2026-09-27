"""FraudStream Evidently monitoring & data drift detection module.

Monitors tabular transaction feature distributions against training baseline.
Reuses canonical FEATURE_COLUMNS from src.features.feature_logic.
Uses top-5 discriminatory features (V17, V14, V12, V10, V16) as key anchors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.features.feature_logic import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Explicitly top-5 discriminatory features by effect size:
# V17 (-8.9), V14 (-7.8), V12 (-6.6), V10 (-5.4), V16 (-4.9)
KEY_DISCRIMINATORY_FEATURES: list[str] = ["V17", "V14", "V12", "V10", "V16"]

# Default feature store reference path
DEFAULT_FEATURE_STORE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "feature_store"
    / "features_v1.parquet"
)

# Training split cutoff: Time <= 145,247s (roughly first 227,845 transactions)
TRAIN_SPLIT_ROW_COUNT = 227845


def load_reference_features(
    path: Path | str | None = None,
    split_only: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load reference baseline features from versioned Parquet feature store."""
    target_path = Path(path) if path else DEFAULT_FEATURE_STORE_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Reference feature store not found at {target_path}. "
            "Ensure feature store exists."
        )

    df = pd.read_parquet(target_path, columns=FEATURE_COLUMNS)
    if split_only and len(df) >= TRAIN_SPLIT_ROW_COUNT:
        df = df.iloc[:TRAIN_SPLIT_ROW_COUNT].copy()

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).copy()

    return df


def generate_synthetic_drift(
    data: pd.DataFrame,
    amount_log_shift: float = 2.0,
    v_mean_shift: float = 1.5,
    v_noise_std: float = 1.0,
    features_to_perturb: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data drift by perturbing feature distributions.

    Used to construct rigorous positive tests proving the detector fires on genuine distributional change.
    """
    df_perturbed = data.copy()
    rng = np.random.default_rng(random_state)
    target_features = features_to_perturb or KEY_DISCRIMINATORY_FEATURES

    # Perturb log_amount if present
    if "log_amount" in df_perturbed.columns:
        df_perturbed["log_amount"] = df_perturbed["log_amount"] + amount_log_shift

    # Perturb discriminatory V-features with mean shift and noise
    for feat in target_features:
        if feat in df_perturbed.columns:
            noise = rng.normal(loc=v_mean_shift, scale=v_noise_std, size=len(df_perturbed))
            df_perturbed[feat] = df_perturbed[feat] + noise

    return df_perturbed


def detect_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_share_threshold: float = 0.20,
    p_value_threshold: float = 0.05,
    key_features_drift_threshold: int = 2,
    html_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Detect covariate shift between reference and current feature distributions.

    Parameters:
        reference_data: Baseline feature DataFrame (training distribution).
        current_data: Window of recent transactions to test for drift.
        drift_share_threshold: Fraction of features that must drift to trigger alarm (default 0.20 = >6/30).
        p_value_threshold: Significance level for two-sample Kolmogorov-Smirnov test (default 0.05).
        key_features_drift_threshold: Count of top-5 discriminatory features that must drift (default 2).
        html_report_path: Optional path to export Evidently HTML drift report.

    Returns:
        Structured dictionary containing drift decision, metrics, and per-feature diagnostics.
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in reference_data.columns or c not in current_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for drift detection: {missing_cols}")

    drifted_features: list[str] = []
    feature_metrics: dict[str, dict[str, Any]] = {}

    for col in FEATURE_COLUMNS:
        ref_vals = reference_data[col].to_numpy(dtype=float)
        cur_vals = current_data[col].to_numpy(dtype=float)

        # 2-sample Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref_vals, cur_vals)
        w_dist = stats.wasserstein_distance(ref_vals, cur_vals)

        is_drifted = bool(ks_res.pvalue < p_value_threshold)
        if is_drifted:
            drifted_features.append(col)

        feature_metrics[col] = {
            "p_value": float(ks_res.pvalue),
            "ks_statistic": float(ks_res.statistic),
            "wasserstein_distance": float(w_dist),
            "drifted": is_drifted,
            "ref_mean": float(np.mean(ref_vals)),
            "cur_mean": float(np.mean(cur_vals)),
            "ref_std": float(np.std(ref_vals)),
            "cur_std": float(np.std(cur_vals)),
        }

    drift_share = len(drifted_features) / len(FEATURE_COLUMNS)
    key_features_drifted = [f for f in KEY_DISCRIMINATORY_FEATURES if f in drifted_features]

    # Documented, justified threshold logic:
    # Trigger drift if:
    # 1. Total feature drift share >= 20% (> 6 of 30 features), OR
    # 2. Key discriminatory features exhibit concentrated shift (>= 2 of V17, V14, V12, V10, V16)
    drift_by_share = drift_share >= drift_share_threshold
    drift_by_key_features = len(key_features_drifted) >= key_features_drift_threshold
    has_drift = bool(drift_by_share or drift_by_key_features)

    # Optional Evidently HTML report generation
    evidently_report_saved = False
    if html_report_path:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            rep = Report([DataDriftPreset()])
            # Run on subset of columns to keep report focused
            rep_ref = reference_data[FEATURE_COLUMNS]
            rep_cur = current_data[FEATURE_COLUMNS]
            rep_result = rep.run(current_data=rep_cur, reference_data=rep_ref)

            out_path = Path(html_report_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rep_result.save_html(str(out_path))
            evidently_report_saved = True
            logger.info("Evidently HTML drift report saved to %s", out_path)
        except Exception as e:
            logger.warning("Failed to generate Evidently HTML report: %s", e)

    return {
        "drift_detected": has_drift,
        "drift_share": float(drift_share),
        "drift_share_threshold": float(drift_share_threshold),
        "drifted_features_count": len(drifted_features),
        "total_features_count": len(FEATURE_COLUMNS),
        "drifted_features": drifted_features,
        "key_features_drifted": key_features_drifted,
        "key_discriminatory_features": KEY_DISCRIMINATORY_FEATURES,
        "drift_by_share": bool(drift_by_share),
        "drift_by_key_features": bool(drift_by_key_features),
        "feature_metrics": feature_metrics,
        "evidently_report_saved": evidently_report_saved,
        "html_report_path": str(html_report_path) if html_report_path else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("[DRIFT DETECTOR] Loading baseline reference features...")
    ref_df = load_reference_features(sample_size=5000)

    # Use hold-out slice for current distribution comparison
    all_df = pd.read_parquet(DEFAULT_FEATURE_STORE_PATH, columns=FEATURE_COLUMNS)
    cur_df = all_df.iloc[TRAIN_SPLIT_ROW_COUNT:].sample(
        n=min(5000, len(all_df) - TRAIN_SPLIT_ROW_COUNT), random_state=42
    )

    print("[DRIFT DETECTOR] Running two-sample drift evaluation...")
    report_output = Path(__file__).resolve().parent.parent.parent / "reports" / "data_drift_report.html"
    drift_result = detect_feature_drift(ref_df, cur_df, html_report_path=report_output)

    print(f"\n--- Drift Evaluation Summary ---")
    print(f"Drift Detected: {drift_result['drift_detected']}")
    print(f"Drift Share: {drift_result['drift_share']:.2%} ({drift_result['drifted_features_count']}/{drift_result['total_features_count']} features)")
    print(f"Key Features Drifted: {drift_result['key_features_drifted']}")
    if drift_result['html_report_path']:
        print(f"Evidently Report Saved: {drift_result['html_report_path']}")

