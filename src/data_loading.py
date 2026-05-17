"""Data loading utilities for the PowerCo churn project.

All loaders accept an optional ``data_dir`` so the same code runs locally and
in CI. Raw filenames match the Forage simulation package; cleaned filenames
match the artefacts written by the EDA / feature-engineering notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project root resolves to the repo root from src/<this file>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"


def _resolve(data_dir: str | Path | None) -> Path:
    """Return a Path for ``data_dir``, defaulting to ``data/`` at repo root."""
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def load_raw(
    data_dir: str | Path | None = None,
    client_file: str = "client_data.csv",
    price_file: str = "price_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw Forage CSVs.

    Returns
    -------
    (client_data, price_data)
    """
    base = _resolve(data_dir)
    client = pd.read_csv(base / client_file)
    price = pd.read_csv(base / price_file)
    # The literal string 'MISSING' encodes nulls in the raw data; convert to NaN.
    for col in ("channel_sales", "origin_up"):
        if col in client.columns:
            client[col] = client[col].replace("MISSING", pd.NA)
    return client, price


def load_model_dataset(
    data_dir: str | Path | None = None,
    filename: str = "data_for_predictions.csv",
) -> pd.DataFrame:
    """Load the feature-engineered dataset used for modelling."""
    return pd.read_csv(_resolve(data_dir) / filename)


def split_features_target(
    df: pd.DataFrame,
    target: str = "churn",
    drop: tuple[str, ...] = ("id", "Unnamed: 0"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into ``(X, y)`` and drop identifier / index columns if present."""
    y = df[target].astype(int)
    X = df.drop(columns=[c for c in (*drop, target) if c in df.columns])
    return X, y
