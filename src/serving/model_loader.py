"""Model and threshold loading for FraudStream serving."""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import xgboost as xgb

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# Track active model source globally for observability (/health endpoint)
_active_model_source: str = "unknown"


def get_active_model_source() -> str:
    """Return the active model source ('registry' or 'local_fallback')."""
    return _active_model_source


def _get_mlflow_uri() -> str:
    db_path = PROJECT_ROOT / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def load_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    prefer_registry: bool = True,
    model_name: str = "fraudstream-xgboost",
    tracking_uri: str | None = None,
) -> xgb.XGBClassifier:
    """Load XGBoost model.

    Attempts to load the 'Production' model from MLflow Model Registry first.
    If registry is unavailable or has no Production model, falls back to local file
    artifact with an explicit warning (never silent fallback).
    """
    global _active_model_source

    custom_dir_specified = Path(model_dir).resolve() != DEFAULT_MODEL_DIR.resolve()

    if prefer_registry and not custom_dir_specified:
        uri = tracking_uri or _get_mlflow_uri()
        try:
            client = MlflowClient(tracking_uri=uri)
            prod_versions = [
                mv for mv in client.search_model_versions(f"name='{model_name}'")
                if mv.current_stage == "Production"
            ]
            if prod_versions:
                mlflow.set_tracking_uri(uri)
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.xgboost.load_model(model_uri)
                _active_model_source = "registry"
                logger.info(
                    "Successfully loaded Production model from MLflow Registry: %s (version %s)",
                    model_name,
                    prod_versions[0].version,
                )
                return model
            else:
                reason = f"No version of model '{model_name}' currently in 'Production' stage"
        except Exception as e:
            reason = f"Registry lookup failed: {e}"

        # If we reached here, registry was preferred but unavailable or had no production model
        warning_msg = (
            f"[WARNING] MLflow Registry fallback triggered: {reason}. "
            f"Falling back to local file artifact in '{model_dir}'."
        )
        logger.warning(warning_msg)
        print(warning_msg)

    # Local fallback path
    model_path = Path(model_dir) / "xgboost_baseline.json"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training pipeline first."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    _active_model_source = "local_fallback"
    return model


def load_threshold_config(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    """Load threshold config from JSON. Fails fast if file missing."""
    config_path = Path(model_dir) / "threshold_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Threshold config not found: {config_path}. Run training pipeline first."
        )
    with open(config_path) as f:
        config = json.load(f)
    if "optimal_threshold" not in config:
        raise ValueError("threshold_config.json missing 'optimal_threshold' key")
    return config
