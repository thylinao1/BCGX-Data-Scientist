"""Tests for src.time_split: cohort split by a monotone column."""

import numpy as np
import pandas as pd
import pytest

from src.time_split import cohort_split, cohort_summary


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
