"""Tests for the categorization cascade (cache -> rules -> mocked LLM)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai.categorize as categorize_module
from ai.cache import MerchantCache
from ai.categorize import Categorization, categorize_keys
from ai.config import MODEL_CATEGORIZER
from ai.privacy import build_categorization_payload
from ai.rules import RuleStore


def item(merchant: str, category: str, confidence: float) -> dict:
    return {"merchant": merchant, "category": category, "confidence": confidence}


def make_response(items: list[dict]):
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    block = SimpleNamespace(type="text", text=json.dumps({"items": items}, ensure_ascii=False))
    return SimpleNamespace(content=[block], usage=usage)


class FakeClient:
    """Client Anthropic simulé : rejoue une liste de réponses ou d'exceptions."""

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


@pytest.fixture
def stores(tmp_path: Path) -> tuple[MerchantCache, RuleStore]:
    return MerchantCache(path=tmp_path / "cache.json"), RuleStore(path=tmp_path / "rules.json")


def test_cache_tier_no_api_call(stores) -> None:
    cache, rules = stores
    cache.set("MCDONALD'S", "Restaurant", 0.9, "llm")
    client = FakeClient([])
    results = categorize_keys(["MCDONALD'S"], cache, rules, client=client)
    assert results["MCDONALD'S"].category == "Restaurant"
    assert results["MCDONALD'S"].source == "cache"
    assert client.calls == []


def test_rule_tier_no_api_call(stores) -> None:
    cache, rules = stores
    rules.add("METRO", "Épicerie", source="legacy")
    client = FakeClient([])
    results = categorize_keys(["METRO ETS"], cache, rules, client=client)
    assert results["METRO ETS"] == Categorization("Épicerie", 1.0, "rule")
    assert client.calls == []


def test_llm_tier_caches_results(stores) -> None:
    cache, rules = stores
    client = FakeClient([make_response([item("NETFLIX.COM", "Abonnements", 0.95)])])
    results = categorize_keys(["NETFLIX.COM"], cache, rules, client=client)
    res = results["NETFLIX.COM"]
    assert (res.category, res.source, res.needs_review) == ("Abonnements", "llm", False)
    assert cache.get("NETFLIX.COM")["category"] == "Abonnements"

    # Second run is a cache hit: no new API call needed
    again = categorize_keys(["NETFLIX.COM"], cache, rules, client=FakeClient([]))
    assert again["NETFLIX.COM"].source == "cache"


def test_low_confidence_flags_review(stores) -> None:
    cache, rules = stores
    client = FakeClient([make_response([item("MYSTERY SHOP", "Autre", 0.3)])])
    results = categorize_keys(["MYSTERY SHOP"], cache, rules, client=client)
    assert results["MYSTERY SHOP"].needs_review is True
    assert cache.get("MYSTERY SHOP")["needs_review"] is True


def test_payload_is_built_by_privacy_guard(stores) -> None:
    cache, rules = stores
    client = FakeClient([make_response([item("UBER TRIP", "Transport", 0.9)])])
    categorize_keys(["UBER TRIP"], cache, rules, client=client)
    call = client.calls[0]
    assert call["model"] == MODEL_CATEGORIZER
    assert call["messages"][0]["content"] == build_categorization_payload(["UBER TRIP"])


def test_invalid_category_retried_once(stores) -> None:
    cache, rules = stores
    client = FakeClient(
        [
            make_response([item("SHOPX", "NotACategory", 0.9)]),
            make_response([item("SHOPX", "Achats", 0.8)]),
        ]
    )
    results = categorize_keys(["SHOPX"], cache, rules, client=client)
    assert len(client.calls) == 2
    assert results["SHOPX"].category == "Achats"


def test_malformed_json_twice_falls_back_to_autre(stores) -> None:
    cache, rules = stores
    bad = SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")], usage=None)
    client = FakeClient([bad, bad])
    results = categorize_keys(["SHOPX"], cache, rules, client=client)
    assert results["SHOPX"] == Categorization("Autre", 0.0, "llm", True)
    # A real model verdict failure is cached so it lands in the review queue
    assert cache.get("SHOPX")["category"] == "Autre"


def test_omitted_merchant_gets_review_flag(stores) -> None:
    cache, rules = stores
    client = FakeClient(
        [
            make_response([item("A SHOP", "Achats", 0.9)]),  # B SHOP missing -> retry once
            make_response([item("A SHOP", "Achats", 0.9)]),  # still missing
        ]
    )
    results = categorize_keys(["A SHOP", "B SHOP"], cache, rules, client=client)
    assert results["A SHOP"].category == "Achats"
    assert results["B SHOP"] == Categorization("Autre", 0.0, "llm", True)
    assert len(client.calls) == 2


def test_api_failure_not_cached(stores) -> None:
    cache, rules = stores
    client = FakeClient([RuntimeError("boom")])
    results = categorize_keys(["SHOPX"], cache, rules, client=client)
    assert results["SHOPX"].source == "fallback"
    assert results["SHOPX"].needs_review is True
    assert cache.get("SHOPX") is None  # retried next run when the API is back


def test_use_llm_false_never_calls_api(stores) -> None:
    cache, rules = stores
    client = FakeClient([])
    results = categorize_keys(["SHOPX"], cache, rules, client=client, use_llm=False)
    assert results["SHOPX"].source == "fallback"
    assert client.calls == []


def test_batching_splits_requests(stores, monkeypatch) -> None:
    cache, rules = stores
    monkeypatch.setattr(categorize_module, "CATEGORIZER_BATCH_SIZE", 2)
    client = FakeClient(
        [
            make_response([item("K1", "Autre", 0.9), item("K2", "Autre", 0.9)]),
            make_response([item("K3", "Autre", 0.9)]),
        ]
    )
    results = categorize_keys(["K1", "K2", "K3"], cache, rules, client=client)
    assert len(client.calls) == 2
    assert all(results[k].source == "llm" for k in ("K1", "K2", "K3"))


def test_duplicate_and_empty_keys_handled(stores) -> None:
    cache, rules = stores
    client = FakeClient([make_response([item("SHOPX", "Achats", 0.9)])])
    results = categorize_keys(["SHOPX", "SHOPX", ""], cache, rules, client=client)
    assert len(client.calls) == 1
    assert list(results) == ["SHOPX"]
