"""Cost-sensitive evaluation utilities.

The PowerCo churn problem is framed as a single GBP-denominated decision
problem. Three business inputs are enough to define every per-instance cost
contribution:

    clv             : customer-lifetime value lost when a churner is missed
    campaign_cost   : cost of contacting one customer with a retention offer
    retention_rate  : probability that a contacted churner is actually retained

The implied per-confusion-matrix-cell contributions are:

    FN  ->  clv                                      (lose the customer)
    FP  ->  campaign_cost                             (waste a contact)
    TP  ->  campaign_cost - clv * retention_rate     (pay to contact; in expectation, save the customer)
    TN  ->  0

With these definitions the two helpers below are mathematically consistent:
``expected_cost(...) == -expected_value(...)`` whenever they are called with
matching parameter values. Selecting an operating threshold by minimising
``expected_cost`` is identical to selecting it by maximising
``expected_value``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CostMatrix:
    """Business-cost parameters for the churn decision problem.

    All per-cell costs are derived from three primary inputs. Override any of
    them to run sensitivity analysis on the corresponding lever.
    """

    clv: float = 50_000.0
    campaign_cost: float = 500.0
    retention_rate: float = 0.3

    @property
    def cost_fn(self) -> float:
        """Cost of missing a churner: the full CLV."""
        return self.clv

    @property
    def cost_fp(self) -> float:
        """Cost of a false alarm: one wasted retention contact."""
        return self.campaign_cost

    @property
    def cost_tp(self) -> float:
        """Net cost of a true positive: pay to contact, recover CLV with probability ``retention_rate``."""
        return self.campaign_cost - self.clv * self.retention_rate

    @property
    def cost_tn(self) -> float:
        """No cost for correctly leaving a non-churner alone."""
        return 0.0


def confusion_counts(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` as plain Python ints."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def expected_cost(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
    costs: CostMatrix = CostMatrix(),
) -> float:
    """Total expected GBP-cost at the supplied operating threshold.

    Negative values mean the model is net-positive versus doing nothing.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return (
        tn * costs.cost_tn
        + fp * costs.cost_fp
        + fn * costs.cost_fn
        + tp * costs.cost_tp
    )


def optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    costs: CostMatrix = CostMatrix(),
    grid: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Sweep a threshold grid and return ``(threshold*, cost*)``.

    The grid should be evaluated on the validation fold, not the test fold.
    """
    if grid is None:
        grid = np.arange(0.01, 0.99, 0.01)
    grid = np.asarray(list(grid))
    costs_arr = np.array([expected_cost(y_true, y_prob, t, costs) for t in grid])
    idx = int(np.argmin(costs_arr))
    return float(grid[idx]), float(costs_arr[idx])


@dataclass
class ThresholdSweep:
    """Output container for :func:`sweep_thresholds`."""

    thresholds: np.ndarray
    costs: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    extras: dict = field(default_factory=dict)


def sweep_thresholds(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    costs: CostMatrix = CostMatrix(),
    grid: Iterable[float] | None = None,
) -> ThresholdSweep:
    """Vectorised threshold sweep returning cost / recall / precision per grid point."""
    if grid is None:
        grid = np.arange(0.01, 0.99, 0.01)
    grid = np.asarray(list(grid), dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    cost_arr = np.zeros_like(grid)
    rec = np.zeros_like(grid)
    prec = np.zeros_like(grid)

    for i, t in enumerate(grid):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        cost_arr[i] = (
            tn * costs.cost_tn
            + fp * costs.cost_fp
            + fn * costs.cost_fn
            + tp * costs.cost_tp
        )
        rec[i] = tp / (tp + fn) if (tp + fn) else 0.0
        prec[i] = tp / (tp + fp) if (tp + fp) else 0.0

    return ThresholdSweep(thresholds=grid, costs=cost_arr, recall=rec, precision=prec)


def expected_value(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
    clv: float = 50_000.0,
    retention_rate: float = 0.3,
    campaign_cost: float = 500.0,
) -> float:
    """Net expected value of the policy at ``threshold``.

    Equivalent to ``-expected_cost(...)`` with matching parameters.
    Kept as a separate entry point so sensitivity sweeps can vary
    ``retention_rate`` directly without constructing a :class:`CostMatrix`.
    """
    return -expected_cost(
        y_true,
        y_prob,
        threshold,
        costs=CostMatrix(
            clv=clv,
            campaign_cost=campaign_cost,
            retention_rate=retention_rate,
        ),
    )
