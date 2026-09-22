"""Lightweight versioned Parquet feature store with JSON manifest tracking."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feature_store"
MANIFEST_NAME = "manifest.json"


def _manifest_path(store_dir: Path) -> Path:
    return store_dir / MANIFEST_NAME


def get_manifest(store_dir: str | Path = DEFAULT_STORE_DIR) -> dict[str, Any]:
    """Read feature store manifest or return empty structure if not found."""
    path = _manifest_path(Path(store_dir))
    if not path.is_file():
        return {"current_version": None, "versions": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict[str, Any], store_dir: Path) -> None:
    path = _manifest_path(store_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_new_version(
    df: pd.DataFrame,
    description: str,
    store_dir: str | Path = DEFAULT_STORE_DIR,
) -> int:
    """Persist feature DataFrame as a new versioned Parquet file without overwriting old versions."""
    base_dir = Path(store_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    manifest = get_manifest(base_dir)
    existing_versions = [int(v) for v in manifest["versions"].keys()]
    next_v = max(existing_versions) + 1 if existing_versions else 1

    filename = f"features_v{next_v}.parquet"
    filepath = base_dir / filename
    df.to_parquet(filepath, index=False)

    manifest["current_version"] = next_v
    manifest["versions"][str(next_v)] = {
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
    }

    _save_manifest(manifest, base_dir)
    return next_v


def get_current_features(store_dir: str | Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    """Load current active feature version from the store."""
    base_dir = Path(store_dir)
    manifest = get_manifest(base_dir)
    curr_v = manifest.get("current_version")
    if curr_v is None:
        raise FileNotFoundError(f"No feature versions registered in store at {base_dir}")
    return get_version(curr_v, base_dir)


def get_version(version_num: int, store_dir: str | Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    """Load a specific historical feature version by version number."""
    base_dir = Path(store_dir)
    manifest = get_manifest(base_dir)
    v_info = manifest.get("versions", {}).get(str(version_num))
    if not v_info:
        raise FileNotFoundError(f"Version {version_num} not found in manifest at {base_dir}")
    filepath = base_dir / v_info["filename"]
    return pd.read_parquet(filepath)


def list_versions(store_dir: str | Path = DEFAULT_STORE_DIR) -> dict[str, Any]:
    """Return dictionary of all registered feature versions and metadata."""
    return get_manifest(Path(store_dir)).get("versions", {})
