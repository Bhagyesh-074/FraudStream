"""Feature matrix construction and chronological train/test split."""

import pandas as pd
from src.features.feature_logic import compute_features, FEATURE_COLUMNS as ALL_FEATURE_COLUMNS
from src.pipeline.cleaning import prepare_features, get_feature_columns


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X (30 columns) and target vector y from dataset using shared logic."""
    return compute_features(df), df["Class"]



def time_based_split(
    df: pd.DataFrame, train_fraction: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform chronological train/test split sorted by Time column."""
    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute scale_pos_weight (legit/fraud ratio) from training split."""
    n_fraud = int((y_train == 1).sum())
    if n_fraud == 0:
        raise ValueError("No fraud samples in training set — cannot compute scale_pos_weight.")
    return float((y_train == 0).sum() / n_fraud)


def split_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Generate train/test split report with class counts and size warnings."""
    tr_f, tr_l = int((train_df["Class"] == 1).sum()), int((train_df["Class"] == 0).sum())
    te_f, te_l = int((test_df["Class"] == 1).sum()), int((test_df["Class"] == 0).sum())

    lines = [
        "TIME-BASED TRAIN/TEST SPLIT REPORT", "=" * 60, "",
        f"Train set:  {len(train_df):>10,} rows  |  Legit: {tr_l:>10,}  |  Fraud: {tr_f:>6,} ({tr_f/len(train_df)*100:.4f}%)  |  Ratio: {tr_l/tr_f:.1f}:1",
        f"Test set:   {len(test_df):>10,} rows  |  Legit: {te_l:>10,}  |  Fraud: {te_f:>6,} ({te_f/len(test_df)*100:.4f}%)  |  Ratio: {te_l/te_f if te_f else 0:.1f}:1",
        f"Time range — Train: [{train_df['Time'].min():.0f}, {train_df['Time'].max():.0f}] | Test: [{test_df['Time'].min():.0f}, {test_df['Time'].max():.0f}]",
    ]
    if te_f < 80:
        lines.extend(["", f"*** WARNING: LOW FRAUD COUNT IN TEST SET ({te_f} < 80) ***"])
    return "\n".join(lines)

