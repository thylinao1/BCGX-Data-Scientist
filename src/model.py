"""Modelling helpers: pipelines, cross-validation, threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from .evaluation import CostMatrix, optimal_threshold


def make_smote_rf(
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_split: int = 5,
) -> ImbPipeline:
    """SMOTE-inside-pipeline + Random Forest.

    Putting SMOTE inside :class:`imblearn.pipeline.Pipeline` ensures that
    synthetic samples are generated **inside each CV fold**, never on data the
    fold's evaluation set has seen.
    """
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train / validation / test split, stratified on ``y``.

    The validation fold is used only for threshold selection; the test fold is
    used only for the final, reported metrics.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@dataclass
class CVResult:
    """Mean / std cost-optimal threshold + recall across CV folds."""

    threshold_mean: float
    threshold_std: float
    recall_mean: float
    recall_std: float
    cost_mean: float
    cost_std: float
    fold_thresholds: list[float]
    fold_costs: list[float]


def cv_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    costs: CostMatrix = CostMatrix(),
    random_state: int = 42,
) -> CVResult:
    """Stratified k-fold; SMOTE inside the pipeline; cost-optimal threshold per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds, fold_costs, recalls = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = make_smote_rf(random_state=random_state)
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        t_star, c_star = optimal_threshold(y_val.values, y_prob, costs)
        thresholds.append(t_star)
        fold_costs.append(c_star)
        # recall at fold-optimal threshold
        y_pred = (y_prob >= t_star).astype(int)
        tp = int(((y_pred == 1) & (y_val.values == 1)).sum())
        fn = int(((y_pred == 0) & (y_val.values == 1)).sum())
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return CVResult(
        threshold_mean=float(np.mean(thresholds)),
        threshold_std=float(np.std(thresholds)),
        recall_mean=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        cost_mean=float(np.mean(fold_costs)),
        cost_std=float(np.std(fold_costs)),
        fold_thresholds=thresholds,
        fold_costs=fold_costs,
    )
