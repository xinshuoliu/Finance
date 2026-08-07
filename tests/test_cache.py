"""Tests for the persistent merchant categorization cache."""

import json
from pathlib import Path

from ai.cache import MerchantCache


def make_cache(tmp_path: Path) -> MerchantCache:
    return MerchantCache(path=tmp_path / "merchant_cache.json")


def test_get_missing_returns_none(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    assert cache.get("UNKNOWN") is None


def test_set_and_get_roundtrip(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.set("AMZN MKTP CA", "Achats", 0.93, "llm")
    entry = cache.get("AMZN MKTP CA")
    assert entry["category"] == "Achats"
    assert entry["confidence"] == 0.93
    assert entry["source"] == "llm"
    assert entry["updated_at"]


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "merchant_cache.json"
    first = MerchantCache(path=path)
    first.set("PHARMACIE JEAN COUTU", "Santé", 1.0, "user")
    first.save()

    second = MerchantCache(path=path)
    assert second.get("PHARMACIE JEAN COUTU")["category"] == "Santé"


def test_accents_readable_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "merchant_cache.json"
    cache = MerchantCache(path=path)
    cache.set("METRO", "Épicerie", 0.9, "llm")
    cache.save()
    assert "Épicerie" in path.read_text(encoding="utf-8")  # ensure_ascii=False


def test_corrupt_file_recovers_empty(tmp_path: Path) -> None:
    path = tmp_path / "merchant_cache.json"
    path.write_text("{not valid json", encoding="utf-8")

    cache = MerchantCache(path=path)
    assert len(cache) == 0

    cache.set("X", "Autre", 0.0, "llm")
    cache.save()
    assert json.loads(path.read_text(encoding="utf-8"))["X"]["category"] == "Autre"


def test_needs_review_flag(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.set("VAGUE SHOP", "Autre", 0.4, "llm", needs_review=True)
    cache.set("CLEAR SHOP", "Épicerie", 0.95, "llm")
    assert cache.get("VAGUE SHOP")["needs_review"] is True
    assert "needs_review" not in cache.get("CLEAR SHOP")


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "cache.json"
    cache = MerchantCache(path=path)
    cache.set("K", "Autre", 0.5, "llm")
    cache.save()
    assert path.exists()


def test_contains_and_len(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    assert "K" not in cache
    cache.set("K", "Autre", 0.5, "llm")
    assert "K" in cache
    assert len(cache) == 1
