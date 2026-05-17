"""Probability calibration helpers.

SMOTE rebalances the training class, but the resulting classifier's predicted
probabilities are no longer calibrated to the true (imbalanced) base rate.
Wrapping the SMOTE + Random Forest pipeline in
:class:`sklearn.calibration.CalibratedClassifierCV` brings the scores back to
the natural prevalence, so the cost-optimal threshold lives on an
interpretable probability scale rather than reflecting an artefact of the
resampling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .model import make_smote_rf


def fit_calibrated_smote_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: Literal["isotonic", "sigmoid"] = "isotonic",
    cv: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Fit a calibrated SMOTE + RF pipeline.

    Calibration uses internal k-fold CV: each fold fits a fresh SMOTE + RF
    pipeline on (k-1)/k of the data and calibrates against the held-out 1/k.
    Final ``predict_proba`` averages over the ``cv`` calibrators.

    Parameters
    ----------
    X_train, y_train
        Training fold. The validation and test folds must be held out.
    method
        ``'isotonic'`` is non-parametric and usually preferred when n is large
        enough; ``'sigmoid'`` (Platt) is two-parameter and lighter-weight.
    cv
        Number of internal folds for calibration. Five is the sklearn default.
    """
    base = make_smote_rf(random_state=random_state)
    calibrated = CalibratedClassifierCV(estimator=base, cv=cv, method=method)
    calibrated.fit(X_train, y_train)
    return calibrated


def calibration_curve_points(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical calibration curve: per-bin mean predicted vs. mean observed.

    Returns ``(mean_predicted, mean_observed, bin_count)``. Useful for a
    reliability-diagram plot in the notebook.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_pred = np.zeros(n_bins)
    mean_obs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            mean_pred[b] = float(y_prob[mask].mean())
            mean_obs[b] = float(y_true[mask].mean())
            counts[b] = int(mask.sum())
    return mean_pred, mean_obs, counts
