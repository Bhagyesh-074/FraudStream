"""Cost matrix for fraud detection evaluation."""

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class CostMatrix:
    """Business cost matrix for evaluating fraud classification outcomes."""

    fn_cost: float | None = None
    fp_cost: float = 10.0
    tp_cost: float = 0.0
    tn_cost: float = 0.0
    avg_fraud_amount: float | None = None

    def calibrate_from_data(self, df: pd.DataFrame) -> "CostMatrix":
        """Set FN cost from average fraud transaction amount in dataset."""
        self.avg_fraud_amount = float(df.loc[df["Class"] == 1, "Amount"].mean())
        self.fn_cost = self.avg_fraud_amount
        return self

    @property
    def cost_ratio(self) -> float:
        """Ratio of FN cost to FP cost."""
        if self.fn_cost is None:
            raise ValueError("Cost matrix not fully initialized. Call calibrate_from_data() first.")
        if self.fp_cost == 0:
            raise ValueError("FP cost cannot be zero (division by zero).")
        return self.fn_cost / self.fp_cost

    def compute_expected_cost(self, tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
        """Compute total business cost for a confusion matrix."""
        if self.fn_cost is None:
            raise ValueError("FN cost not set. Call calibrate_from_data() first.")
        tp_tot, fp_tot = tp * self.tp_cost, fp * self.fp_cost
        fn_tot, tn_tot = fn * self.fn_cost, tn * self.tn_cost
        return {
            "tp_cost": tp_tot, "fp_cost": fp_tot, "fn_cost": fn_tot, "tn_cost": tn_tot,
            "total_cost": tp_tot + fp_tot + fn_tot + tn_tot,
            "tp_count": tp, "fp_count": fp, "fn_count": fn, "tn_count": tn,
        }

    def summary(self) -> dict[str, Any]:
        """Return cost matrix values as a dictionary."""
        return {
            "fn_cost": self.fn_cost, "fp_cost": self.fp_cost,
            "tp_cost": self.tp_cost, "tn_cost": self.tn_cost,
            "avg_fraud_amount": self.avg_fraud_amount,
            "cost_ratio": self.cost_ratio if self.fn_cost is not None else None,
        }

    def __repr__(self) -> str:
        ratio = f"{self.cost_ratio:.1f}:1" if self.fn_cost is not None else "N/A"
        return f"CostMatrix(FN=${self.fn_cost:.2f}, FP=${self.fp_cost:.2f}, ratio={ratio})"

