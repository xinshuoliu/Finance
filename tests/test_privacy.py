"""Tests for the privacy guard: no amount can ever reach the API."""

import json

import pandas as pd
import pytest

from ai.privacy import build_categorization_payload


def test_roundtrip_with_accents() -> None:
    keys = ["MCDONALD'S", "ÉPICERIE MÉTRO"]
    payload = build_categorization_payload(keys)
    assert json.loads(payload) == keys
    assert "ÉPICERIE" in payload  # ensure_ascii=False


def test_rejects_numbers() -> None:
    with pytest.raises(TypeError):
        build_categorization_payload(["OK", 123.45])
    with pytest.raises(TypeError):
        build_categorization_payload([100])


def test_rejects_transaction_records() -> None:
    with pytest.raises(TypeError):
        build_categorization_payload([{"Details": "SHOP", "Amount": 99.99}])


def test_rejects_amount_series_and_rows() -> None:
    df = pd.DataFrame({"Details": ["SHOP A", "SHOP B"], "Amount": [12.34, 999.99]})
    with pytest.raises(TypeError):
        build_categorization_payload(df["Amount"])  # numeric amounts can never reach the API
    with pytest.raises(TypeError):
        build_categorization_payload(list(df.itertuples()))


def test_rejects_bare_string() -> None:
    with pytest.raises(TypeError):
        build_categorization_payload("SHOP")


def test_payload_carries_no_transaction_data() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-15", "2026-01-16"]),
            "Details": ["SHOP A", "SHOP B"],
            "Amount": [123.45, 678.90],
        }
    )
    payload = build_categorization_payload(df["Details"].tolist())
    assert "123.45" not in payload
    assert "678.9" not in payload
    assert "2026-01-15" not in payload
