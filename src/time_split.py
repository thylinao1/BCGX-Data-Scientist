"""Cohort-based train/test split by a monotone column, plus a pseudo
out-of-time evaluation of the cost-sensitive policy.

The PowerCo dataset is a single 2015 snapshot of churn outcomes, so a true
out-of-time hold-out is not possible: every customer's churn label is observed
at the same calendar time. The closest defensible substitute is to partition
customers by ``date_activ`` (contract activation date): the earliest-activated
customers train, the most-recently-activated customers form a pseudo
out-of-time test fold.

This is not a substitute for genuine out-of-time validation, because all
churn labels are still observed at the same calendar moment. What it does test
is whether a model trained on older contract cohorts still discriminates, and
whether the cost-optimal threshold still holds, when it is applied to the most
recently acquired customers. On this dataset the later cohorts churn more (the
churn rate climbs from roughly 8 percent in the 2009 cohorts to roughly 15
percent in the 2013 cohorts), so the split is informative rather than cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

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


@dataclass
class OutOfTimeReport:
    """Result of :func:`out_of_time_cost_eval`.

    ``threshold`` is selected on a validation slice of the *training* cohort
    only; every figure suffixed ``_test`` is measured on the held-out
    later-activation cohort, which the model and the threshold never saw.
    """

    n_train: int
    n_test: int
    train_churn_rate: float
    test_churn_rate: float
    threshold: float
    test_auc: float
    recall_at_threshold: float
    recall_at_half: float
    cost_at_threshold: float
    cost_at_half: float
    cost_reduction: float  # cost_at_half - cost_at_threshold
    cutoff: object  # the date_activ (or sort_col) value used to split


def out_of_time_cost_eval(
    df: pd.DataFrame,
    date_col: str = "date_activ",
    target: str = "churn",
    feature_cols: Sequence[str] | None = None,
    test_quantile: float = 0.8,
    costs=None,
    random_state: int = 42,
) -> OutOfTimeReport:
    """Train on the earlier-activation cohort, score the later one.

    A SMOTE + Random Forest pipeline is fit on the earliest ``test_quantile``
    fraction of customers (by ``date_col``). The cost-optimal threshold is
    selected on a validation slice carved out of that training cohort, so the
    later-activation cohort is genuinely held out. The cost matrix is then
    evaluated on the later cohort to test whether the threshold and the cost
    reduction survive temporal drift.

    Parameters
    ----------
    df
        Modelling frame containing ``date_col``, ``target`` and the features.
    date_col
        Monotone column to split the cohort on (contract activation date).
    feature_cols
        Columns to use as model inputs. Defaults to every column except
        ``date_col``, ``target`` and obvious identifier columns.
    test_quantile
        Fraction of customers (earliest by ``date_col``) used for training.
    costs
        A :class:`evaluation.CostMatrix`. Defaults to the project parameters.

    Returns
    -------
    OutOfTimeReport
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from .evaluation import CostMatrix, expected_cost, optimal_threshold
    from .model import make_smote_rf

    if costs is None:
        costs = CostMatrix()

    train_df, test_df = cohort_split(
        df, sort_col=date_col, test_quantile=test_quantile, test_is_above=True
    )
    cutoff = train_df[date_col].max()

    if feature_cols is None:
        exclude = {date_col, target, "id", "Unnamed: 0"}
        feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = list(feature_cols)

    X_train_all = train_df[feature_cols].astype(float)
    y_train_all = train_df[target].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target].astype(int)

    # Threshold is tuned on a validation slice of the training cohort only.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_all,
    )
    pipe = make_smote_rf(random_state=random_state)
    pipe.fit(X_tr, y_tr)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    threshold, _ = optimal_threshold(y_val.to_numpy(), val_prob, costs)

    # Everything below is measured on the held-out later-activation cohort.
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_prob))
    cost_threshold = expected_cost(y_test.to_numpy(), test_prob, threshold, costs)
    cost_half = expected_cost(y_test.to_numpy(), test_prob, 0.5, costs)

    def _recall(t: float) -> float:
        pred = (test_prob >= t).astype(int)
        truth = y_test.to_numpy()
        tp = int(((pred == 1) & (truth == 1)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        return tp / (tp + fn) if (tp + fn) else 0.0

    return OutOfTimeReport(
        n_train=len(train_df),
        n_test=len(test_df),
        train_churn_rate=float(y_train_all.mean()),
        test_churn_rate=float(y_test.mean()),
        threshold=threshold,
        test_auc=test_auc,
        recall_at_threshold=_recall(threshold),
        recall_at_half=_recall(0.5),
        cost_at_threshold=float(cost_threshold),
        cost_at_half=float(cost_half),
        cost_reduction=float(cost_half - cost_threshold),
        cutoff=cutoff,
    )
