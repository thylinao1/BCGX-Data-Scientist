"""Feature engineering for the PowerCo churn dataset.

These functions are intentionally pure (no I/O, no global state) so they can
be unit-tested and reused inside scikit-learn pipelines.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DATE_COLUMNS: tuple[str, ...] = (
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal",
)


def parse_dates(df: pd.DataFrame, columns: Iterable[str] = DATE_COLUMNS) -> pd.DataFrame:
    """Coerce date columns in-place. Returns the same frame for chaining."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def add_temporal_features(
    df: pd.DataFrame,
    reference_date: str | pd.Timestamp = "2016-01-01",
) -> pd.DataFrame:
    """Add tenure / time-to-end / month-of-event features.

    ``reference_date`` is treated as the modelling snapshot ("today").
    """
    out = parse_dates(df)
    ref = pd.to_datetime(reference_date)

    if "date_activ" in out.columns:
        out["tenure_days"] = (ref - out["date_activ"]).dt.days
    if "date_end" in out.columns:
        out["days_to_end"] = (out["date_end"] - ref).dt.days
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[f"month_{col.replace('date_', '')}"] = out[col].dt.month
    return out


def add_price_volatility(df: pd.DataFrame, price_cols: Iterable[str]) -> pd.DataFrame:
    """Per-row mean / max / variance over a set of price-change columns."""
    out = df.copy()
    cols = [c for c in price_cols if c in out.columns]
    if not cols:
        return out
    out["avg_price_change_year"] = out[cols].mean(axis=1)
    out["max_price_change_year"] = out[cols].max(axis=1)
    out["var_price_change_year"] = out[cols].var(axis=1)
    return out


def add_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    """Margin-per-kWh and consumption-change-rate, with safe denominators."""
    out = df.copy()
    if {"net_margin", "cons_12m"}.issubset(out.columns):
        out["margin_per_kwh"] = out["net_margin"] / (out["cons_12m"] + 1)
    if {"cons_last_month", "cons_12m"}.issubset(out.columns):
        out["cons_change_rate"] = (out["cons_last_month"] * 12) / (out["cons_12m"] + 1)
    return out


def log_transform_skewed(
    df: pd.DataFrame,
    columns: Iterable[str],
    skew_threshold: float = 1.0,
) -> pd.DataFrame:
    """Apply ``log1p`` to columns whose skewness exceeds ``skew_threshold``.

    Negative values are clipped at zero before the transform to avoid NaNs.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if out[col].dropna().skew() > skew_threshold:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def build_feature_frame(
    df: pd.DataFrame,
    price_change_cols: Iterable[str] = (),
    reference_date: str | pd.Timestamp = "2016-01-01",
) -> pd.DataFrame:
    """Convenience: run the whole feature pipeline in one call."""
    out = add_temporal_features(df, reference_date=reference_date)
    out = add_price_volatility(out, price_change_cols)
    out = add_consumption_features(out)
    return out
