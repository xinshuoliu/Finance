"""Tests for the correction flow: corrections must survive a re-import."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.cache import MerchantCache
from ai.categorize import categorize_keys, record_user_correction
from ai.rules import RuleStore


def make_response(items: list[dict]):
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    block = SimpleNamespace(type="text", text=json.dumps({"items": items}, ensure_ascii=False))
    return SimpleNamespace(content=[block], usage=usage)


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "cache.json", tmp_path / "rules.json"


def test_correction_survives_reimport(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)

    # First import: the AI gets it wrong with low confidence
    client = FakeClient(
        [make_response([{"merchant": "SUPER GYM", "category": "Autre", "confidence": 0.4}])]
    )
    first = categorize_keys(["SUPER GYM"], cache, rules, client=client)
    assert first["SUPER GYM"].category == "Autre"
    assert first["SUPER GYM"].needs_review is True

    # The user corrects it (what the Review tab does)
    record_user_correction("SUPER GYM", "Loisirs", cache, rules)

    # Re-import of the same statement, in a brand-new session, with the API
    # offline: the correction must still hold
    cache2 = MerchantCache(path=cache_path)
    rules2 = RuleStore(path=rules_path)
    again = categorize_keys(["SUPER GYM"], cache2, rules2, client=None, use_llm=False)
    assert again["SUPER GYM"].category == "Loisirs"
    assert again["SUPER GYM"].source == "cache"
    assert again["SUPER GYM"].needs_review is False
    assert again["SUPER GYM"].confidence == 1.0


def test_correction_applies_to_new_variants_of_the_merchant(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)

    record_user_correction("SUPER GYM", "Loisirs", cache, rules)

    # A future statement produces a slightly different key for the same
    # merchant: the rule (substring match) still catches it, no API needed
    result = categorize_keys(["SUPER GYM PLUS"], cache, rules, client=None, use_llm=False)
    assert result["SUPER GYM PLUS"].category == "Loisirs"
    assert result["SUPER GYM PLUS"].source == "rule"


def test_correction_beats_stale_llm_cache(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)
    cache.set("SQDC", "Achats", 0.9, "llm")

    record_user_correction("SQDC", "Santé", cache, rules)

    result = categorize_keys(["SQDC"], cache, rules, client=None, use_llm=False)
    assert result["SQDC"].category == "Santé"
    assert result["SQDC"].source == "cache"


def test_correction_writes_rule_with_user_source(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)

    record_user_correction("SUPER GYM", "Loisirs", cache, rules)

    saved = json.loads(rules_path.read_text(encoding="utf-8"))
    assert saved[0]["pattern"] == "SUPER GYM"
    assert saved[0]["category"] == "Loisirs"
    assert saved[0]["source"] == "user"
    assert saved[0]["created"]


def test_empty_key_is_rejected(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)
    assert record_user_correction("", "Loisirs", cache, rules) is False
    assert len(rules) == 0


def test_batched_corrections_save_once(paths) -> None:
    cache_path, rules_path = paths
    cache = MerchantCache(path=cache_path)
    rules = RuleStore(path=rules_path)

    record_user_correction("SHOP A", "Achats", cache, rules, save=False)
    record_user_correction("SHOP B", "Restaurant", cache, rules, save=False)
    assert not rules_path.exists()  # nothing written yet

    rules.save()
    cache.save()
    assert len(json.loads(rules_path.read_text(encoding="utf-8"))) == 2
    assert json.loads(cache_path.read_text(encoding="utf-8"))["SHOP B"]["category"] == "Restaurant"
