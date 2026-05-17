"""Cox and Kaplan-Meier helpers.

The module:

* log-transforms heavy-tailed covariates before fitting, so coefficients are
  interpretable as percentage changes in the hazard;
* reports the concordance index alongside the summary;
* runs the Schoenfeld residual test of the proportional-hazards assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# lifelines is imported lazily inside functions so importing this module is
# cheap and the tests don't need it on the path.


HEAVY_TAILED_COVARIATES: tuple[str, ...] = ("cons_12m", "net_margin")


def prepare_cox_frame(
    df: pd.DataFrame,
    covariates: Iterable[str],
    duration_col: str = "tenure",
    event_col: str = "churn",
    log_cols: Iterable[str] = HEAVY_TAILED_COVARIATES,
) -> pd.DataFrame:
    """Return a dropna'd frame ready for ``CoxPHFitter.fit``.

    Heavy-tailed columns are ``log1p``-transformed (after clipping at zero) so
    their coefficients are interpretable as percentage changes in the hazard.
    """
    keep = [duration_col, event_col, *covariates]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy().dropna()
    for col in log_cols:
        if col in out.columns:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


@dataclass
class CoxReport:
    """Container for a fitted Cox model plus diagnostic test results."""

    summary: pd.DataFrame
    concordance: float
    schoenfeld: pd.DataFrame  # p-values per covariate (from proportional_hazard_test)
    ph_assumption_holds: bool


def fit_cox(
    df: pd.DataFrame,
    duration_col: str = "tenure",
    event_col: str = "churn",
    penalizer: float = 0.01,
    significance: float = 0.05,
) -> CoxReport:
    """Fit a Cox PH model and run the proportional-hazards diagnostic.

    Parameters
    ----------
    penalizer
        L2 penalty on the partial-likelihood. 0.01 is a mild ridge; the
        original notebook used 0.1 with no justification.
    significance
        Cut-off for the Schoenfeld residual test. The PH assumption is taken
        to hold if **all** covariate p-values exceed this threshold.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    ph_result = proportional_hazard_test(cph, df, time_transform="rank")
    schoenfeld = ph_result.summary
    holds = bool((schoenfeld["p"] > significance).all())

    return CoxReport(
        summary=cph.summary,
        concordance=float(cph.concordance_index_),
        schoenfeld=schoenfeld,
        ph_assumption_holds=holds,
    )
