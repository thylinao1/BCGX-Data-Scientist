"""Tests for src.uncertainty: bootstrap confidence intervals."""

import numpy as np
import pytest

from src.evaluation import CostMatrix
from src.uncertainty import BootstrapResult, bootstrap_metric


def test_bootstrap_result_contains_samples(synthetic_probs):
    y, p = synthetic_probs
    res = bootstrap_metric(y, p, threshold=0.5, metric="recall", n_bootstrap=200)
    assert isinstance(res, BootstrapResult)
    assert res.samples.shape == (200,)


def test_bootstrap_recall_ci_contains_point_estimate(synthetic_probs):
    y, p = synthetic_probs
    res = bootstrap_metric(y, p, threshold=0.3, metric="recall", n_bootstrap=500)
    # The percentile CI must straddle the point estimate.
    assert res.ci_lo <= res.point_estimate <= res.ci_hi


def test_bootstrap_precision_returns_valid_range(synthetic_probs):
    y, p = synthetic_probs
    res = bootstrap_metric(y, p, threshold=0.3, metric="precision", n_bootstrap=300)
    assert 0.0 <= res.ci_lo <= res.ci_hi <= 1.0


def test_bootstrap_expected_cost_uses_cost_matrix(synthetic_probs):
    y, p = synthetic_probs
    cm = CostMatrix(clv=100_000, campaign_cost=200, retention_rate=0.4)
    res = bootstrap_metric(
        y, p, threshold=0.1, metric="expected_cost",
        costs=cm, n_bootstrap=300,
    )
    # With heavy FN penalty the costs should be large in magnitude.
    assert np.abs(res.point_estimate) > 0
    assert res.ci_lo <= res.ci_hi


def test_invalid_confidence_raises(synthetic_probs):
    y, p = synthetic_probs
    with pytest.raises(ValueError):
        bootstrap_metric(y, p, threshold=0.5, confidence=1.5)


def test_invalid_metric_raises(synthetic_probs):
    y, p = synthetic_probs
    with pytest.raises(ValueError):
        bootstrap_metric(y, p, threshold=0.5, metric="nonsense", n_bootstrap=10)  # type: ignore[arg-type]
