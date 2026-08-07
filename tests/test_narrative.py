"""Tests for the monthly narrative: summary math and number grounding."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ai.config import MODEL_NARRATIVE
from ai.narrative import (
    NarrativeError,
    build_month_summary,
    check_numbers,
    generate_narrative,
)
from ai.privacy import build_narrative_payload
from analysis import detect_recurring


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


def text_response(text: str):
    usage = SimpleNamespace(
        input_tokens=500, output_tokens=120, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)


@pytest.fixture
def txns() -> pd.DataFrame:
    rows = [
        # January
        ("2026-01-05", "POKE MONSTER", "Restaurant", 30.0, "Debit"),
        ("2026-01-20", "MCDONALD'S", "Restaurant", 20.0, "Debit"),
        ("2026-01-10", "METRO", "Épicerie", 80.0, "Debit"),
        # February
        ("2026-02-08", "MCDONALD'S", "Restaurant", 100.0, "Debit"),
        ("2026-02-12", "METRO", "Épicerie", 40.0, "Debit"),
        ("2026-02-15", "NEW SHOP", "Achats", 60.0, "Debit"),
        ("2026-02-28", "EMPLOYER PAY", "Revenu", 2000.0, "Credit"),
    ]
    dates, merchants, categories, amounts, kinds = zip(*rows)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(list(dates)),
            "Details": list(merchants),
            "Merchant": list(merchants),
            "Amount": list(amounts),
            "Debit/Credit": list(kinds),
            "Category": list(categories),
        }
    )


# ---------- build_month_summary ----------


def test_summary_by_category_and_total(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    assert summary["month"] == "2026-02"
    assert summary["total_spent"] == 200.0
    assert summary["transaction_count"] == 3
    assert summary["by_category"] == {"Restaurant": 100.0, "Achats": 60.0, "Épicerie": 40.0}


def test_summary_relative_deltas(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    assert summary["vs_prev_month_pct"]["Restaurant"] == 100.0  # 50 -> 100
    assert summary["vs_prev_month_pct"]["Épicerie"] == -50.0  # 80 -> 40
    assert "Achats" not in summary["vs_prev_month_pct"]  # no previous spending


def test_summary_new_merchants(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    assert summary["top_new_merchants"] == [{"merchant": "NEW SHOP", "amount": 60.0}]
    # First month of the file: "new" is meaningless without history
    assert build_month_summary(txns, "2026-01")["top_new_merchants"] == []


def test_summary_budget_status(txns) -> None:
    summary = build_month_summary(txns, "2026-02", budgets={"Restaurant": 80.0, "Santé": 0.0})
    assert summary["budget_status"] == {"Restaurant": {"budget": 80.0, "spent": 100.0}}


def test_summary_includes_recurring(txns) -> None:
    recurring = pd.DataFrame(
        [
            {
                "Merchant": "NETFLIX.COM",
                "Frequency": "monthly",
                "Occurrences": 3,
                "AverageAmount": 16.99,
                "LastDate": pd.Timestamp("2026-02-15"),
                "NextExpected": pd.Timestamp("2026-03-17"),
                "MonthlyEstimate": 16.99,
            }
        ]
    )
    summary = build_month_summary(txns, "2026-02", recurring=recurring)
    assert summary["recurring_detected"] == [
        {"merchant": "NETFLIX.COM", "frequency": "monthly", "average_amount": 16.99}
    ]
    # The whole summary must pass the privacy guard (plain JSON values only)
    json.loads(build_narrative_payload(summary))


# ---------- privacy guard ----------


def test_narrative_payload_rejects_dataframes(txns) -> None:
    with pytest.raises(TypeError):
        build_narrative_payload({"transactions": txns})
    with pytest.raises(TypeError):
        build_narrative_payload({"dates": pd.Timestamp("2026-01-01")})
    with pytest.raises(TypeError):
        build_narrative_payload([1, 2, 3])


# ---------- number grounding ----------


def test_check_numbers_accepts_grounded_text(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    text = (
        "In February 2026 you spent 200.00 CAD in total. Restaurant was the biggest "
        "category at 100.00 CAD, up 100.0% from January, while Épicerie dropped 50.0%."
    )
    assert check_numbers(text, summary) == []


def test_check_numbers_flags_invented_numbers(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    text = "You spent about 999.99 CAD, including 123 CAD on snacks."
    assert check_numbers(text, summary) == ["999.99", "123"]


def test_check_numbers_handles_thousands_separator() -> None:
    summary = {"month": "2026-02", "total_spent": 1234.56}
    assert check_numbers("You spent 1,234.56 CAD.", summary) == []


def test_check_numbers_allows_list_counts() -> None:
    summary = {"recurring_detected": [{"merchant": "A"}, {"merchant": "B"}]}
    assert check_numbers("You have 2 active subscriptions.", summary) == []


# ---------- generate_narrative ----------


def test_generate_narrative_happy_path(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    client = FakeClient([text_response("You spent 200.0 CAD, mostly on Restaurant (100.0 CAD).")])
    result = generate_narrative(summary, client=client)
    assert result.suspect_numbers == []
    call = client.calls[0]
    assert call["model"] == MODEL_NARRATIVE
    assert call["messages"][0]["content"] == build_narrative_payload(summary)


def test_generate_narrative_reports_suspects(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    client = FakeClient([text_response("You spent exactly 777.77 CAD this month.")])
    result = generate_narrative(summary, client=client)
    assert result.suspect_numbers == ["777.77"]


def test_generate_narrative_api_failure(txns) -> None:
    summary = build_month_summary(txns, "2026-02")
    client = FakeClient([RuntimeError("boom")])
    with pytest.raises(NarrativeError, match="request failed"):
        generate_narrative(summary, client=client)


def test_generate_narrative_without_key(monkeypatch, txns) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(NarrativeError, match="not configured"):
        generate_narrative(build_month_summary(txns, "2026-02"))


def test_detected_recurring_feeds_summary() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15"]),
            "Details": ["NETFLIX.COM"] * 3,
            "Merchant": ["NETFLIX.COM"] * 3,
            "Amount": [16.99] * 3,
            "Debit/Credit": ["Debit"] * 3,
            "Category": ["Abonnements"] * 3,
        }
    )
    summary = build_month_summary(df, "2026-03", recurring=detect_recurring(df))
    assert summary["recurring_detected"][0]["merchant"] == "NETFLIX.COM"
