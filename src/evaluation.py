"""Cost-sensitive evaluation utilities.

The PowerCo churn problem is framed as a £-denominated decision problem:

    expected_cost = FN * cost_fn          # missed churner -> lost CLV
                  + FP * cost_fp          # unnecessary retention contact
                  + TP * cost_tp          # successful retention (negative => benefit)

These helpers are deliberately pure so they can be unit-tested and reused for
CV-style threshold selection without copy-pasting code into notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CostMatrix:
    """Business-cost parameters.

    Defaults mirror the headline scenario in the modelling notebook but every
    sensitivity sweep should override them rather than relying on these values.
    """

    cost_fn: float = 50_000.0   # lost customer-lifetime value
    cost_fp: float = 500.0      # cost of one retention contact
    cost_tp: float = -10_000.0  # net benefit of one successful retention
    cost_tn: float = 0.0


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
    """Total expected £-cost at the supplied operating threshold.

    Negative values mean the model is net-positive vs. doing nothing.
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

    Useful for sensitivity analysis on retention rate / CLV / campaign cost.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    _, fp, fn, tp = confusion_counts(y_true, y_pred)
    benefit = tp * clv * retention_rate
    contact_cost = (tp + fp) * campaign_cost
    missed = fn * clv
    return benefit - contact_cost - missed
