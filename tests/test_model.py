"""Tests for src.model: pipeline construction and split logic."""

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from src.model import make_smote_rf, three_way_split


def test_make_smote_rf_returns_imblearn_pipeline():
    pipe = make_smote_rf()
    assert isinstance(pipe, ImbPipeline)
    step_names = [name for name, _ in pipe.steps]
    # SMOTE must come before the classifier so it is refit per CV fold.
    assert step_names == ["smote", "rf"]


def test_three_way_split_proportions(rng):
    X = pd.DataFrame(rng.normal(size=(1000, 5)))
    y = pd.Series(rng.binomial(1, 0.1, size=1000))
    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(
        X, y, val_size=0.2, test_size=0.2, random_state=0
    )
    n = len(X)
    assert abs(len(X_te) / n - 0.2) < 0.01
    assert abs(len(X_val) / n - 0.2) < 0.01
    assert abs(len(X_tr) / n - 0.6) < 0.01
    # Stratification preserves the class ratio.
    target_ratio = y.mean()
    for fold in (y_tr, y_val, y_te):
        assert abs(fold.mean() - target_ratio) < 0.03


def test_three_way_split_no_overlap(rng):
    X = pd.DataFrame(rng.normal(size=(200, 3)), index=range(200))
    y = pd.Series(rng.binomial(1, 0.3, size=200), index=range(200))
    X_tr, X_val, X_te, *_ = three_way_split(X, y, random_state=1)
    train_idx = set(X_tr.index)
    val_idx = set(X_val.index)
    test_idx = set(X_te.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
