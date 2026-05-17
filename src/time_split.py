"""Cohort-based train/test split by a monotone column.

The PowerCo dataset is a single 2015 snapshot of churn outcomes, so a true
out-of-time hold-out is not possible: every customer's churn label is observed
at the same calendar time. The closest defensible substitute is a cohort
split on either ``date_activ`` (activation date) or ``tenure`` (years on book
at snapshot time). Earlier-activated customers train, more-recently-activated
customers test.

A cohort split tests cross-cohort generalisation. It does not test temporal
generalisation of the churn outcome itself. The notebook calls this out
explicitly so the result is not over-claimed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def cohort_split(
    df: pd.DataFrame,
    sort_col: str = "date_activ",
    test_quantile: float = 0.8,
    test_is_above: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split customers by quantile of a monotone column.

    Parameters
    ----------
    df
        Input frame containing ``sort_col``.
    sort_col
        Column to sort the cohort on. Can be a datetime column (e.g.
        ``date_activ``) or a numeric column whose values increase with later
        activation (e.g. negative ``tenure``).
    test_quantile
        Quantile cutoff. Customers at or below the cutoff form one fold;
        those strictly above form the other.
    test_is_above
        If ``True`` the *higher* values go into the test fold (the default for
        ``date_activ``: later dates are more recent). If ``False`` the lower
        values go into the test fold (useful for ``tenure``, where lower
        values mean more recent activation).

    Returns
    -------
    (train, test)
        Two dataframes with disjoint row indices.
    """
    if sort_col not in df.columns:
        raise KeyError(f"sort_col {sort_col!r} not in dataframe")
    if not 0.0 < test_quantile < 1.0:
        raise ValueError("test_quantile must be in (0, 1)")

    out = df.copy()
    col = out[sort_col]
    if is_datetime64_any_dtype(col):
        col = pd.to_datetime(col, errors="coerce")
    else:
        col = pd.to_numeric(col, errors="coerce")
    out[sort_col] = col

    if col.isna().all():
        raise ValueError(f"{sort_col} parses entirely to NaN; cannot split")

    cutoff = col.quantile(test_quantile)
    if test_is_above:
        train = out[col <= cutoff].copy()
        test = out[col > cutoff].copy()
    else:
        train = out[col >= cutoff].copy()
        test = out[col < cutoff].copy()
    return train, test


def cohort_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sort_col: str = "date_activ",
) -> pd.DataFrame:
    """Quick descriptive summary of the two cohorts for the notebook."""
    rows = []
    for name, fold in [("train", train), ("test", test)]:
        col = fold[sort_col]
        if is_datetime64_any_dtype(col):
            col = pd.to_datetime(col, errors="coerce")
        else:
            col = pd.to_numeric(col, errors="coerce")
        rows.append(
            {
                "fold": name,
                "n": len(fold),
                "churn_rate": float(fold["churn"].mean()) if "churn" in fold.columns else np.nan,
                f"{sort_col}_min": col.min(),
                f"{sort_col}_max": col.max(),
            }
        )
    return pd.DataFrame(rows)
