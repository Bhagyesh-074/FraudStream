"""Cost matrix for fraud detection evaluation."""

import pandas as pd
import numpy as np
from typing import Any


class CostMatrix:
    """Business cost matrix for evaluating fraud classification outcomes."""

    DEFAULT_FP_COST = 10.0

    def __init__(
        self,
        fn_cost: float | None = None,
        fp_cost: float = DEFAULT_FP_COST,
        tp_cost: float = 0.0,
        tn_cost: float = 0.0,
        avg_fraud_amount: float | None = None,
    ):
        """Initialize cost matrix parameters."""
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.avg_fraud_amount = avg_fraud_amount

    def calibrate_from_data(self, df: pd.DataFrame) -> "CostMatrix":
        """Set FN cost from average fraud transaction amount in dataset."""
        fraud_amounts = df.loc[df["Class"] == 1, "Amount"]
        self.avg_fraud_amount = float(fraud_amounts.mean())
        self.fn_cost = self.avg_fraud_amount
        return self

    @property
    def cost_ratio(self) -> float:
        """Ratio of FN cost to FP cost."""
        if self.fn_cost is None or self.fp_cost is None:
            raise ValueError(
                "Cost matrix not fully initialized. Call calibrate_from_data() first."
            )
        if self.fp_cost == 0:
            raise ValueError("FP cost cannot be zero (division by zero).")
        return self.fn_cost / self.fp_cost

    def compute_expected_cost(
        self, tp: int, fp: int, fn: int, tn: int
    ) -> dict[str, float]:
        """Compute total business cost for a confusion matrix."""
        if self.fn_cost is None:
            raise ValueError(
                "FN cost not set. Call calibrate_from_data() first."
            )

        tp_total = tp * self.tp_cost
        fp_total = fp * self.fp_cost
        fn_total = fn * self.fn_cost
        tn_total = tn * self.tn_cost
        total = tp_total + fp_total + fn_total + tn_total

        return {
            "tp_cost": tp_total,
            "fp_cost": fp_total,
            "fn_cost": fn_total,
            "tn_cost": tn_total,
            "total_cost": total,
            "tp_count": tp,
            "fp_count": fp,
            "fn_count": fn,
            "tn_count": tn,
        }

    def summary(self) -> dict[str, Any]:
        """Return cost matrix values as a dictionary."""
        return {
            "fn_cost": self.fn_cost,
            "fp_cost": self.fp_cost,
            "tp_cost": self.tp_cost,
            "tn_cost": self.tn_cost,
            "avg_fraud_amount": self.avg_fraud_amount,
            "cost_ratio": self.cost_ratio if self.fn_cost is not None else None,
        }

    def __repr__(self) -> str:
        ratio_str = f"{self.cost_ratio:.1f}" if self.fn_cost is not None else "N/A"
        return (
            f"CostMatrix(\n"
            f"  FN (missed fraud)   = ${self.fn_cost:.2f}\n"
            f"  FP (false alarm)    = ${self.fp_cost:.2f}\n"
            f"  TP (caught fraud)   = ${self.tp_cost:.2f}\n"
            f"  TN (passed legit)   = ${self.tn_cost:.2f}\n"
            f"  Cost ratio (FN/FP)  = {ratio_str}:1\n"
            f"  Avg fraud amount    = ${self.avg_fraud_amount:.2f}\n"
            f")"
        )
