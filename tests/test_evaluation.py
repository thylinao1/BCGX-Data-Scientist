"""Tests for src.evaluation: cost-sensitive decision logic."""

import numpy as np
import pytest

from src.evaluation import (
    CostMatrix,
    confusion_counts,
    expected_cost,
    expected_value,
    optimal_threshold,
    sweep_thresholds,
)


def test_confusion_counts_simple():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_expected_cost_matches_manual_calculation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    # threshold 0.5 -> y_pred = [0, 1, 0, 1] -> tn=1, fp=1, fn=1, tp=1
    costs = CostMatrix(cost_fn=50_000, cost_fp=500, cost_tp=-10_000, cost_tn=0)
    expected = 1 * 0 + 1 * 500 + 1 * 50_000 + 1 * (-10_000)
    assert expected_cost(y_true, y_prob, threshold=0.5, costs=costs) == expected


def test_optimal_threshold_minimises_cost(synthetic_probs):
    y, p = synthetic_probs
    t_star, c_star = optimal_threshold(y, p)
    # The chosen threshold must beat the naive 0.5 threshold.
    assert c_star <= expected_cost(y, p, 0.5)
    assert 0.0 < t_star < 1.0


def test_sweep_thresholds_returns_monotonic_recall(synthetic_probs):
    y, p = synthetic_probs
    res = sweep_thresholds(y, p)
    # Recall is non-increasing in threshold by construction.
    assert (np.diff(res.recall) <= 1e-9).all()
    assert res.thresholds.shape == res.costs.shape == res.recall.shape


def test_expected_value_negative_when_retention_rate_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    # If retention_rate == 0 we get no benefit but still pay campaign + missed costs.
    ev = expected_value(y, p, threshold=0.5, retention_rate=0.0)
    assert ev < 0


@pytest.mark.parametrize("rate,sign", [(0.0, -1), (0.9, +1)])
def test_expected_value_monotonic_in_retention_rate(synthetic_probs, rate, sign):
    y, p = synthetic_probs
    ev = expected_value(y, p, threshold=0.05, retention_rate=rate)
    assert np.sign(ev) == sign or ev == 0
