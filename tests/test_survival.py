"""Tests for src.survival: the Random Survival Forest helpers.

scikit-survival is an optional heavy dependency. The whole module is skipped
if it is not importable, so a developer without it can still run the rest of
the suite.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

from src.survival import (  # noqa: E402
    DEFAULT_COVARIATES,
    SurvivalReport,
    evaluate_survival_forest,
    make_survival_target,
    prepare_survival_frame,
)


def _make_survival_frame(rng, n=600):
    """A small frame with three covariates plus tenure / churn.

    The event probability is tied to ``net_margin`` so the model has a real
    signal to find and the concordance index lands above chance.
    """
    net_margin = rng.normal(loc=200, scale=60, size=n)
    cons_12m = rng.lognormal(mean=10, sigma=1.2, size=n)
    has_gas = rng.integers(0, 2, size=n)
    # lower-margin customers churn sooner
    churn_p = 1 / (1 + np.exp((net_margin - 200) / 40))
    churn = rng.binomial(1, churn_p)
    tenure = rng.integers(1, 12, size=n)
    return pd.DataFrame(
        {
            "cons_12m": cons_12m,
            "net_margin": net_margin,
            "has_gas": has_gas,
            "tenure": tenure,
            "churn": churn,
        }
    )


def test_make_survival_target_fields(rng):
    df = _make_survival_frame(rng, n=100)
    y = make_survival_target(df)
    assert set(y.dtype.names) == {"event", "time"}
    assert y["event"].dtype == bool
    assert len(y) == len(df)
    assert np.array_equal(y["time"], df["tenure"].to_numpy().astype(float))


def test_prepare_survival_frame_excludes_duration_and_event(rng):
    df = _make_survival_frame(rng, n=120)
    # explicitly include tenure / churn in the covariate list; they must be dropped
    X, y = prepare_survival_frame(
        df, covariates=["cons_12m", "net_margin", "tenure", "churn"]
    )
    assert "tenure" not in X.columns
    assert "churn" not in X.columns
    assert list(X.columns) == ["cons_12m", "net_margin"]
    assert len(X) == len(y)


def test_prepare_survival_frame_filters_missing_covariates(rng):
    df = _make_survival_frame(rng, n=80)
    # only some DEFAULT_COVARIATES exist in this small frame
    X, _ = prepare_survival_frame(df, covariates=DEFAULT_COVARIATES)
    assert set(X.columns).issubset(set(DEFAULT_COVARIATES))
    assert "cons_12m" in X.columns and "net_margin" in X.columns


def test_prepare_survival_frame_raises_when_no_covariates(rng):
    df = _make_survival_frame(rng, n=50)
    with pytest.raises(ValueError):
        prepare_survival_frame(df, covariates=["does_not_exist"])


def test_prepare_survival_frame_drops_missing_rows(rng):
    df = _make_survival_frame(rng, n=100)
    df.loc[df.index[:10], "net_margin"] = np.nan
    X, y = prepare_survival_frame(df, covariates=["cons_12m", "net_margin"])
    assert len(X) == 90
    assert len(y) == 90


def test_evaluate_survival_forest_returns_report(rng):
    df = _make_survival_frame(rng, n=600)
    report = evaluate_survival_forest(
        df,
        covariates=["cons_12m", "net_margin", "has_gas"],
        n_estimators=30,
        n_repeats=3,
        random_state=0,
    )
    assert isinstance(report, SurvivalReport)
    assert 0.0 <= report.concordance_test <= 1.0
    assert 0.0 <= report.concordance_train <= 1.0
    assert report.n_covariates == 3
    assert list(report.importance.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
    ]
    assert len(report.importance) == 3


def test_evaluate_survival_forest_beats_chance_with_signal(rng):
    """With a genuine margin -> churn signal the held-out c-index clears 0.5."""
    df = _make_survival_frame(rng, n=900)
    report = evaluate_survival_forest(
        df,
        covariates=["cons_12m", "net_margin", "has_gas"],
        n_estimators=60,
        n_repeats=3,
        random_state=0,
    )
    assert report.concordance_test > 0.55
