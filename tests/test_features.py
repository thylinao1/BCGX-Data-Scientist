"""Tests for src.features: pure functions, fast to run."""

import numpy as np
import pandas as pd

from src.features import (
    add_consumption_features,
    add_price_volatility,
    add_temporal_features,
    log_transform_skewed,
    parse_dates,
)


def test_parse_dates_coerces_iso_strings():
    df = pd.DataFrame({"date_activ": ["2015-01-01", "not-a-date"]})
    out = parse_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date_activ"])
    assert pd.isna(out["date_activ"].iloc[1])


def test_add_temporal_features_creates_expected_columns(synthetic_client_frame):
    out = add_temporal_features(synthetic_client_frame)
    assert "tenure_days" in out.columns
    assert "days_to_end" in out.columns
    assert "month_activ" in out.columns
    # tenure_days should be non-negative for activations before reference_date
    assert (out["tenure_days"] >= 0).all()


def test_add_price_volatility_handles_missing_cols(synthetic_client_frame):
    out = add_price_volatility(synthetic_client_frame, price_cols=["does_not_exist"])
    # No new columns should be added if none of the requested cols are present.
    assert "avg_price_change_year" not in out.columns


def test_add_price_volatility_computes_stats():
    df = pd.DataFrame({"p1": [1.0, 2.0], "p2": [3.0, 4.0]})
    out = add_price_volatility(df, price_cols=["p1", "p2"])
    assert np.allclose(out["avg_price_change_year"], [2.0, 3.0])
    assert np.allclose(out["max_price_change_year"], [3.0, 4.0])


def test_add_consumption_features_safe_division():
    df = pd.DataFrame(
        {"net_margin": [10.0, 20.0], "cons_12m": [0.0, 99.0], "cons_last_month": [0.0, 8.0]}
    )
    out = add_consumption_features(df)
    # Denominator is (cons_12m + 1) so it never blows up at zero consumption.
    assert np.isfinite(out["margin_per_kwh"]).all()
    assert np.isfinite(out["cons_change_rate"]).all()


def test_log_transform_skewed_only_transforms_skewed(rng):
    df = pd.DataFrame(
        {
            "skewed": rng.lognormal(mean=2, sigma=2, size=500),
            "symmetric": rng.normal(loc=0, scale=1, size=500),
        }
    )
    before = df.copy()
    out = log_transform_skewed(df, columns=["skewed", "symmetric"])
    # Symmetric column should be untouched.
    assert np.allclose(out["symmetric"].values, before["symmetric"].values)
    # Skewed column should be transformed (values smaller in magnitude on average).
    assert out["skewed"].max() < before["skewed"].max()
