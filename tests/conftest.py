"""Shared pytest fixtures.

The Forage data is not redistributable, so all tests run against synthetic
frames that match the schema of the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_client_frame(rng):
    """Mimics the cleaned PowerCo client frame at small scale."""
    n = 500
    start = pd.Timestamp("2010-01-01")
    return pd.DataFrame(
        {
            "id": [f"c{i:04d}" for i in range(n)],
            "cons_12m": rng.lognormal(mean=10, sigma=1.5, size=n),
            "cons_last_month": rng.lognormal(mean=8, sigma=1.5, size=n),
            "net_margin": rng.normal(loc=200, scale=50, size=n),
            "has_gas": rng.integers(0, 2, size=n),
            "tenure": rng.integers(0, 10, size=n),
            "date_activ": start + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D"),
            "date_end": start + pd.to_timedelta(rng.integers(1500, 3000, size=n), unit="D"),
            "date_modif_prod": start + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D"),
            "date_renewal": start + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D"),
            "churn": rng.binomial(1, p=0.1, size=n),
        }
    )


@pytest.fixture
def synthetic_probs(rng):
    """Imbalanced (y_true, y_prob) pair with realistic structure."""
    n = 2000
    y = rng.binomial(1, p=0.1, size=n)
    # Probabilities calibrated to make recall non-trivial.
    p = np.where(y == 1, rng.beta(5, 5, size=n), rng.beta(2, 8, size=n))
    return y, p
