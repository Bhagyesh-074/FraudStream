"""
Dataset download helper for FraudStream.

The Kaggle Credit Card Fraud dataset must be downloaded manually and placed
at data/creditcard.csv. This module provides a check function and instructions.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
from pathlib import Path


# Default path relative to project root
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "creditcard.csv"

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Dataset not found: creditcard.csv                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  This project uses the Kaggle Credit Card Fraud Detection dataset.   ║
║                                                                      ║
║  To download:                                                        ║
║  1. Go to: {KAGGLE_DATASET_URL}                                      ║
║  2. Click "Download" (requires a free Kaggle account)                ║
║  3. Extract the ZIP file                                             ║
║  4. Place 'creditcard.csv' in: data/creditcard.csv                   ║
║                                                                      ║
║  Expected file: creditcard.csv (~150 MB)                             ║
║  Expected shape: ~284,807 rows × 31 columns                         ║
║  Columns: Time, V1-V28, Amount, Class                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def check_dataset_exists(path: Path = DEFAULT_CSV_PATH) -> bool:
    """Check if the dataset file exists at the expected path.

    Returns True if the file exists, False otherwise. When False, prints
    manual download instructions to stdout.
    """
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[OK] Dataset found: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        return False


def get_data_path(path: Path = DEFAULT_CSV_PATH) -> Path:
    """Return the dataset path if it exists, raise FileNotFoundError otherwise."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Download from {KAGGLE_DATASET_URL} and place creditcard.csv "
            f"in the data/ directory."
        )
    return path


if __name__ == "__main__":
    check_dataset_exists()
