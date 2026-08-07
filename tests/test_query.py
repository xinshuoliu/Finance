"""Tests for natural-language query translation and local execution."""

import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from ai.config import MODEL_QUERY
from ai.privacy import build_query_payload
from ai.query import (
    FilterSpec,
    QueryTranslationError,
    describe_spec,
    execute_spec,
    translate_question,
)


def make_response(payload: dict):
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    block = SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))
    return SimpleNamespace(content=[block], usage=usage)


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# ---------- FilterSpec validation ----------


def test_valid_spec_parses() -> None:
    spec = FilterSpec.model_validate(
        {
            "categories": ["Restaurant"],
            "date_from": "2026-01-01",
            "date_to": None,
            "aggregate": "sum",
            "group_by": "month",
            "merchant_contains": None,
            "transaction_type": "debit",
        }
    )
    assert spec.categories == ["Restaurant"]
    assert spec.date_from == date(2026, 1, 1)
    assert spec.group_by == "month"


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown categories: Groceries"):
        FilterSpec.model_validate({"categories": ["Groceries"]})


def test_bad_aggregate_rejected() -> None:
    with pytest.raises(ValidationError):
        FilterSpec.model_validate({"aggregate": "median"})


def test_bad_group_by_rejected() -> None:
    with pytest.raises(ValidationError):
        FilterSpec.model_validate({"group_by": "year"})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        FilterSpec.model_validate({"exec": "import os"})


def test_bad_date_rejected() -> None:
    with pytest.raises(ValidationError):
        FilterSpec.model_validate({"date_from": "janvier"})


def test_dates_out_of_order_rejected() -> None:
    with pytest.raises(ValidationError, match="date_from is after date_to"):
        FilterSpec.model_validate({"date_from": "2026-03-01", "date_to": "2026-01-01"})


# ---------- Local execution ----------


@pytest.fixture
def txns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-01-10", "2026-01-20", "2026-02-05", "2026-02-15", "2026-02-20"]
            ),
            "Details": ["POKE MONSTER", "MCDONALD'S #22028", "METRO", "PAY DEPOSIT", "MCDONALD'S #9"],
            "Merchant": ["POKE MONSTER", "MCDONALD'S", "METRO", "PAY DEPOSIT", "MCDONALD'S"],
            "Amount": [30.0, 20.0, 100.0, 2000.0, 10.0],
            "Debit/Credit": ["Debit", "Debit", "Debit", "Credit", "Debit"],
            "Category": ["Restaurant", "Restaurant", "Épicerie", "Revenu", "Restaurant"],
        }
    )


def test_sum_with_category_filter(txns) -> None:
    spec = FilterSpec(categories=["Restaurant"])
    result = execute_spec(spec, txns)
    assert result.value == 60.0
    assert len(result.filtered) == 3


def test_date_range_filter(txns) -> None:
    spec = FilterSpec(categories=["Restaurant"], date_from=date(2026, 2, 1))
    assert execute_spec(spec, txns).value == 10.0


def test_count_aggregate(txns) -> None:
    spec = FilterSpec(categories=["Restaurant"], aggregate="count")
    result = execute_spec(spec, txns)
    assert result.value == 3
    assert isinstance(result.value, int)


def test_mean_and_max(txns) -> None:
    assert execute_spec(FilterSpec(categories=["Épicerie"], aggregate="mean"), txns).value == 100.0
    assert execute_spec(FilterSpec(aggregate="max"), txns).value == 100.0  # debits only by default


def test_credit_type(txns) -> None:
    spec = FilterSpec(transaction_type="credit", categories=["Revenu"])
    assert execute_spec(spec, txns).value == 2000.0


def test_merchant_contains(txns) -> None:
    spec = FilterSpec(merchant_contains="mcdonald")
    assert execute_spec(spec, txns).value == 30.0


def test_group_by_month(txns) -> None:
    spec = FilterSpec(categories=["Restaurant"], group_by="month")
    table = execute_spec(spec, txns).table
    assert list(table["Month"]) == ["2026-01", "2026-02"]
    assert list(table["Total"]) == [50.0, 10.0]


def test_group_by_category_sorted_desc(txns) -> None:
    table = execute_spec(FilterSpec(group_by="category"), txns).table
    assert list(table["Category"]) == ["Épicerie", "Restaurant"]
    assert list(table["Total"]) == [100.0, 60.0]


def test_group_by_merchant_count(txns) -> None:
    table = execute_spec(FilterSpec(group_by="merchant", aggregate="count"), txns).table
    assert list(table["Merchant"])[0] == "MCDONALD'S"
    assert list(table["Count"])[0] == 2


def test_empty_result(txns) -> None:
    result = execute_spec(FilterSpec(categories=["Santé"]), txns)
    assert result.value == 0.0
    assert result.filtered.empty


def test_describe_spec() -> None:
    spec = FilterSpec(
        categories=["Restaurant"],
        date_from=date(2026, 1, 1),
        group_by="month",
        merchant_contains="poke",
    )
    text = describe_spec(spec)
    assert "Total of expenses" in text
    assert "Restaurant" in text
    assert "from 2026-01-01" in text
    assert 'merchant contains "poke"' in text
    assert "grouped by month" in text


# ---------- Translation via the (mocked) API ----------

VALID_PAYLOAD = {
    "categories": ["Restaurant"],
    "date_from": "2026-01-01",
    "date_to": None,
    "aggregate": "sum",
    "group_by": "month",
    "merchant_contains": None,
    "transaction_type": "debit",
}


def test_translate_returns_validated_spec() -> None:
    client = FakeClient([make_response(VALID_PAYLOAD)])
    spec = translate_question(
        "How much did I spend on restaurants since January?", client=client, today=date(2026, 8, 7)
    )
    assert spec.categories == ["Restaurant"]
    assert spec.date_from == date(2026, 1, 1)

    call = client.calls[0]
    assert call["model"] == MODEL_QUERY
    assert call["messages"][0]["content"] == build_query_payload(
        "How much did I spend on restaurants since January?"
    )
    assert "2026-08-07" in call["system"]  # relative dates resolve against today


def test_translate_invalid_category_raises() -> None:
    bad = dict(VALID_PAYLOAD, categories=["Groceries"])
    client = FakeClient([make_response(bad)])
    with pytest.raises(QueryTranslationError, match="invalid filter"):
        translate_question("groceries?", client=client)


def test_translate_unreadable_answer_raises() -> None:
    block = SimpleNamespace(type="text", text="not json at all")
    client = FakeClient([SimpleNamespace(content=[block], usage=None)])
    with pytest.raises(QueryTranslationError, match="unreadable"):
        translate_question("anything", client=client)


def test_translate_api_failure_raises() -> None:
    client = FakeClient([RuntimeError("boom")])
    with pytest.raises(QueryTranslationError, match="request failed"):
        translate_question("anything", client=client)


def test_translate_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(QueryTranslationError, match="not configured"):
        translate_question("anything")


def test_empty_question_raises() -> None:
    with pytest.raises(QueryTranslationError, match="type a question"):
        translate_question("   ", client=FakeClient([]))


def test_query_payload_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        build_query_payload(123.45)
    assert build_query_payload("  hello ") == "hello"
