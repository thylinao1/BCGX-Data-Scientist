"""Bootstrap uncertainty quantification for headline metrics.

The cross-validated threshold already has a mean and standard deviation, but
recall, precision and the GBP-cost on the test fold are reported as single
point estimates. The helpers below resample the test fold to attach
percentile-based confidence intervals to those numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .evaluation import CostMatrix, confusion_counts, expected_cost


Metric = Literal["recall", "precision", "expected_cost"]


@dataclass
class BootstrapResult:
    """Point estimate plus percentile-based confidence interval."""

    point_estimate: float
    ci_lo: float
    ci_hi: float
    samples: np.ndarray

    def __repr__(self) -> str:
        return (
            f"BootstrapResult(point={self.point_estimate:,.3f}, "
            f"95% CI=[{self.ci_lo:,.3f}, {self.ci_hi:,.3f}])"
        )


def _metric_value(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: Metric,
    costs: CostMatrix,
) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    if metric == "recall":
        return tp / (tp + fn) if (tp + fn) else 0.0
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else 0.0
    if metric == "expected_cost":
        return expected_cost(y_true, y_prob, threshold, costs)
    raise ValueError(f"Unknown metric: {metric!r}")


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: Metric = "recall",
    costs: CostMatrix = CostMatrix(),
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap percentile confidence interval for a confusion-matrix metric.

    Parameters
    ----------
    y_true, y_prob
        True labels and predicted probabilities on the evaluation fold.
    threshold
        Operating threshold to apply when computing the metric.
    metric
        One of ``'recall'``, ``'precision'``, ``'expected_cost'``.
    costs
        Cost matrix; ignored unless ``metric == 'expected_cost'``.
    n_bootstrap
        Number of bootstrap resamples.
    confidence
        Coverage of the returned interval, e.g. 0.95 for a 95% CI.
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    point = _metric_value(y_true, y_prob, threshold, metric, costs)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = _metric_value(y_true[idx], y_prob[idx], threshold, metric, costs)

    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        point_estimate=float(point),
        ci_lo=float(np.quantile(samples, alpha)),
        ci_hi=float(np.quantile(samples, 1.0 - alpha)),
        samples=samples,
    )
