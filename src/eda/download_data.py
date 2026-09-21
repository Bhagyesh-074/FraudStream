"""Dataset verification helper for FraudStream."""

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "creditcard.csv"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
Dataset not found at data/creditcard.csv.
Please download from: {KAGGLE_DATASET_URL}
Extract and place 'creditcard.csv' in the data/ directory.
"""


def check_dataset_exists(path: Path = DEFAULT_CSV_PATH) -> bool:
    """Check if creditcard.csv exists at the target path."""
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[OK] Dataset found: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        return False


def get_data_path(path: Path = DEFAULT_CSV_PATH) -> Path:
    """Return dataset path if file exists, raise FileNotFoundError otherwise."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from {KAGGLE_DATASET_URL}."
        )
    return path


if __name__ == "__main__":
    check_dataset_exists()
