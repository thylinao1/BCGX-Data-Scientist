"""Tests for src.calibration: probability calibration on SMOTE + RF."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.calibration import calibration_curve_points, fit_calibrated_smote_rf


def _make_synthetic(rng, n=600):
    """Imbalanced synthetic dataset that resembles the PowerCo prior."""
    X = pd.DataFrame(rng.normal(size=(n, 5)))
    # Make class 1 a noisy linear function of feature 0 so RF can learn signal.
    logits = 0.8 * X[0].values + rng.normal(scale=0.5, size=n) - 1.8
    y = pd.Series((logits > 0).astype(int))
    return X, y


def test_fit_calibrated_smote_rf_returns_calibrated_object(rng):
    X, y = _make_synthetic(rng)
    model = fit_calibrated_smote_rf(X, y, method="isotonic", cv=3)
    assert isinstance(model, CalibratedClassifierCV)
    # The wrapped fold pipelines must each be SMOTE + RF.
    assert len(model.calibrated_classifiers_) == 3


def test_calibrated_predictions_in_unit_interval(rng):
    X, y = _make_synthetic(rng)
    model = fit_calibrated_smote_rf(X, y, method="sigmoid", cv=3)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert probs.shape == (len(X),)


def test_calibration_curve_points_sums_match_input():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    mean_pred, mean_obs, counts = calibration_curve_points(y_true, y_prob, n_bins=5)
    assert counts.sum() == len(y_true)
    assert mean_pred.shape == mean_obs.shape == counts.shape == (5,)
