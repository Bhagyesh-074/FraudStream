"""FraudStream monitoring package for drift detection and data quality."""

from src.monitoring.drift_detector import (
    KEY_DISCRIMINATORY_FEATURES,
    detect_feature_drift,
    generate_synthetic_drift,
    load_reference_features,
)

__all__ = [
    "KEY_DISCRIMINATORY_FEATURES",
    "detect_feature_drift",
    "generate_synthetic_drift",
    "load_reference_features",
]
