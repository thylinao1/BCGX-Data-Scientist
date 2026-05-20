"""Tests for src.time_split: cohort split and out-of-time cost evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.time_split import (
    OutOfTimeReport,
    cohort_split,
    cohort_summary,
    out_of_time_cost_eval,
)


def _make_dated_frame(rng, n=500):
    base = pd.Timestamp("2010-01-01")
    days = rng.integers(0, 2_000, size=n)
    return pd.DataFrame(
        {
            "date_activ": [base + pd.Timedelta(days=int(d)) for d in days],
            "churn": rng.binomial(1, 0.1, size=n),
            "feature": rng.normal(size=n),
        }
    )


def _make_tenure_frame(rng, n=500):
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 15, size=n),
            "churn": rng.binomial(1, 0.1, size=n),
            "feature": rng.normal(size=n),
        }
    )


def test_cohort_split_returns_disjoint_folds(rng):
    df = _make_dated_frame(rng)
    train, test = cohort_split(df, sort_col="date_activ", test_quantile=0.8)
    train_ids = set(train.index)
    test_ids = set(test.index)
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(df)


def test_cohort_split_train_activates_before_test(rng):
    df = _make_dated_frame(rng)
    train, test = cohort_split(df, sort_col="date_activ", test_quantile=0.8)
    train_max = pd.to_datetime(train["date_activ"]).max()
    test_min = pd.to_datetime(test["date_activ"]).min()
    assert train_max <= test_min


def test_cohort_split_numeric_column(rng):
    df = _make_tenure_frame(rng)
    # Lower tenure = more recent activation, so test_is_above=False.
    train, test = cohort_split(df, sort_col="tenure", test_quantile=0.2, test_is_above=False)
    # Train fold should have the higher tenure values (older customers).
    assert train["tenure"].min() >= test["tenure"].max()


def test_cohort_split_approximate_size(rng):
    df = _make_dated_frame(rng, n=2_000)
    train, test = cohort_split(df, sort_col="date_activ", test_quantile=0.8)
    # Test fold should be roughly 20%.
    assert 0.15 <= len(test) / len(df) <= 0.30


def test_cohort_summary_columns(rng):
    df = _make_dated_frame(rng)
    train, test = cohort_split(df, sort_col="date_activ", test_quantile=0.7)
    summary = cohort_summary(train, test, sort_col="date_activ")
    assert list(summary["fold"]) == ["train", "test"]
    assert {"n", "churn_rate"}.issubset(summary.columns)
    assert "date_activ_min" in summary.columns
    assert "date_activ_max" in summary.columns


def test_cohort_split_missing_col_raises(rng):
    df = pd.DataFrame({"churn": [0, 1, 0]})
    with pytest.raises(KeyError):
        cohort_split(df, sort_col="date_activ")


def test_cohort_split_invalid_quantile_raises(rng):
    df = _make_dated_frame(rng, n=50)
    with pytest.raises(ValueError):
        cohort_split(df, sort_col="date_activ", test_quantile=0.0)
    with pytest.raises(ValueError):
        cohort_split(df, sort_col="date_activ", test_quantile=1.5)


def _make_model_frame(rng, n=800):
    """A modelling frame with date_activ, churn and three numeric features.

    Churn probability is tied to ``margin`` so the SMOTE + RF pipeline has a
    real signal to fit and the cost evaluation produces a sensible threshold.
    """
    base = pd.Timestamp("2009-01-01")
    days = rng.integers(0, 1_800, size=n)
    margin = rng.normal(loc=200, scale=60, size=n)
    cons = rng.lognormal(mean=10, sigma=1.0, size=n)
    price_var = rng.normal(loc=0, scale=1, size=n)
    churn_p = 1 / (1 + np.exp((margin - 170) / 35))
    churn = rng.binomial(1, churn_p)
    return pd.DataFrame(
        {
            "date_activ": [base + pd.Timedelta(days=int(d)) for d in days],
            "margin": margin,
            "cons": cons,
            "price_var": price_var,
            "churn": churn,
        }
    )


def test_out_of_time_cost_eval_returns_report(rng):
    df = _make_model_frame(rng, n=800)
    report = out_of_time_cost_eval(
        df, feature_cols=["margin", "cons", "price_var"], random_state=0
    )
    assert isinstance(report, OutOfTimeReport)
    assert report.n_train + report.n_test == len(df)
    assert 0.0 <= report.test_auc <= 1.0
    assert 0.0 < report.threshold < 1.0


def test_out_of_time_cost_eval_threshold_reduces_cost(rng):
    """The cost-optimal threshold should not be worse than the default 0.5."""
    df = _make_model_frame(rng, n=900)
    report = out_of_time_cost_eval(
        df, feature_cols=["margin", "cons", "price_var"], random_state=0
    )
    assert report.cost_at_threshold <= report.cost_at_half
    assert report.cost_reduction >= 0.0


def test_out_of_time_cost_eval_recall_in_range(rng):
    df = _make_model_frame(rng, n=800)
    report = out_of_time_cost_eval(
        df, feature_cols=["margin", "cons", "price_var"], random_state=0
    )
    assert 0.0 <= report.recall_at_threshold <= 1.0
    assert 0.0 <= report.recall_at_half <= 1.0
    # the lower threshold should never catch fewer churners than 0.5
    assert report.recall_at_threshold >= report.recall_at_half


def test_out_of_time_cost_eval_default_feature_cols(rng):
    """With feature_cols=None, identifier / date / target columns are dropped."""
    df = _make_model_frame(rng, n=700)
    df["id"] = [f"c{i:04d}" for i in range(len(df))]
    report = out_of_time_cost_eval(df, random_state=0)
    assert isinstance(report, OutOfTimeReport)
    assert report.n_test > 0
