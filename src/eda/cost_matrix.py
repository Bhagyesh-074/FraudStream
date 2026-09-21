"""
Cost matrix for fraud detection evaluation.

Defines explicit costs for each cell of the confusion matrix (TP, FP, FN, TN).
This cost matrix drives threshold selection and business-cost evaluation in
Phase 2 and beyond.

Design choices documented inline — the point is to be explicit and defensible,
not perfectly accurate. Real-world cost matrices would be calibrated with the
fraud operations team; these are reasonable starting assumptions.
"""

import pandas as pd
import numpy as np
from typing import Any


class CostMatrix:
    """Business cost matrix for fraud detection.

    Cost semantics:
      - FALSE NEGATIVE (missed fraud): The transaction amount is lost entirely.
        This is the dominant cost in fraud detection — every missed fraud is a
        direct financial loss equal to the transaction value.

      - FALSE POSITIVE (false alarm on legitimate): Customer friction from
        blocking/flagging a legitimate transaction. Includes: support call cost
        (~$5-7 industry average), customer dissatisfaction/churn risk (~$3-5
        estimated lifetime value impact). We use a flat per-incident cost since
        customer friction doesn't scale with transaction amount.

      - TRUE POSITIVE (correctly caught fraud): Small investigation cost but
        net positive (prevented loss). We set this to $0 for simplicity — the
        prevented loss is captured by NOT incurring the FN cost.

      - TRUE NEGATIVE (correctly passed legitimate): No action, no cost.

    The cost ratio (FN/FP) tells us how many false alarms we should tolerate
    to catch one more fraud. If FN_cost=$122 and FP_cost=$10, the ratio is
    ~12:1, meaning we should accept up to 12 false alarms to avoid missing
    one fraud. This directly informs threshold selection.

    Attributes:
        fn_cost: Cost of a false negative (missed fraud). Derived from data.
        fp_cost: Cost of a false positive (false alarm). Assumed flat rate.
        tp_cost: Cost of a true positive (caught fraud). Default 0.
        tn_cost: Cost of a true negative (passed legit). Default 0.
        avg_fraud_amount: Average fraud transaction amount from data.
    """

    # Assumed cost of a false positive (customer friction per incident)
    # Breakdown: ~$5-7 support call + ~$3-5 customer dissatisfaction/churn risk
    DEFAULT_FP_COST = 10.0

    def __init__(
        self,
        fn_cost: float | None = None,
        fp_cost: float = DEFAULT_FP_COST,
        tp_cost: float = 0.0,
        tn_cost: float = 0.0,
        avg_fraud_amount: float | None = None,
    ):
        """Initialize cost matrix.

        Args:
            fn_cost: Cost per false negative. If None, must be set via
                     calibrate_from_data() before use.
            fp_cost: Cost per false positive (default: $10 customer friction).
            tp_cost: Cost per true positive (default: $0).
            tn_cost: Cost per true negative (default: $0).
            avg_fraud_amount: Average fraud amount, stored for reference.
        """
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.avg_fraud_amount = avg_fraud_amount

    def calibrate_from_data(self, df: pd.DataFrame) -> "CostMatrix":
        """Set FN cost from the actual average fraud transaction amount.

        The reasoning: a missed fraud costs the full transaction amount.
        Using the average fraud amount is a reasonable proxy since we don't
        know at prediction time what a specific transaction's amount will be
        if it turns out to be fraud.

        Args:
            df: Dataset with 'Amount' and 'Class' columns.

        Returns:
            self (for chaining).
        """
        fraud_amounts = df.loc[df["Class"] == 1, "Amount"]
        self.avg_fraud_amount = float(fraud_amounts.mean())
        self.fn_cost = self.avg_fraud_amount
        return self

    @property
    def cost_ratio(self) -> float:
        """Ratio of FN cost to FP cost.

        Interpretation: how many false alarms are acceptable to prevent
        one missed fraud. Higher ratio → more aggressive fraud catching
        (lower threshold).

        Raises:
            ValueError: If costs are not yet set.
        """
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
        """Compute total business cost for a given confusion matrix.

        Args:
            tp: True positive count.
            fp: False positive count.
            fn: False negative count.
            tn: True negative count.

        Returns:
            Dict with per-cell costs and total cost.

        Raises:
            ValueError: If FN cost is not set.
        """
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
        """Return cost matrix values as a dict for reporting."""
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
