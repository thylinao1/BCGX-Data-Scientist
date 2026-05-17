"""Tests for src.data_loading: file paths and column hygiene."""

import pandas as pd
import pytest

from src.data_loading import load_model_dataset, load_raw, split_features_target


def test_split_features_target_drops_id_and_target(synthetic_client_frame):
    X, y = split_features_target(synthetic_client_frame)
    assert "id" not in X.columns
    assert "churn" not in X.columns
    assert len(X) == len(y)
    assert y.dtype.kind == "i"


def test_load_raw_resolves_missing_token(tmp_path):
    client = pd.DataFrame(
        {
            "id": ["a", "b"],
            "channel_sales": ["MISSING", "x"],
            "origin_up": ["y", "MISSING"],
        }
    )
    price = pd.DataFrame({"id": ["a"], "price_off_peak": [0.1]})
    client.to_csv(tmp_path / "client_data.csv", index=False)
    price.to_csv(tmp_path / "price_data.csv", index=False)

    c, p = load_raw(data_dir=tmp_path)
    assert c["channel_sales"].isna().sum() == 1
    assert c["origin_up"].isna().sum() == 1
    assert len(p) == 1


def test_load_model_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_dataset(data_dir=tmp_path, filename="does_not_exist.csv")
