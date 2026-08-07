"""Tests des règles de catégorisation déterministes."""

from pathlib import Path

from ai.rules import RuleStore


def make_store(tmp_path: Path) -> RuleStore:
    return RuleStore(path=tmp_path / "rules.json")


def test_substring_match_case_insensitive(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add("netflix", "Abonnements", source="legacy")
    assert rules.match("NETFLIX.COM") == "Abonnements"


def test_no_match_returns_none(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add("METRO", "Épicerie")
    assert rules.match("PHARMAPRIX") is None
    assert rules.match("") is None


def test_user_rules_take_precedence(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add("SP SHOPPAY", "Achats", source="legacy")
    rules.add("SHOPPAY LAURA", "Loisirs", source="user")
    assert rules.match("SP SHOPPAY LAURA") == "Loisirs"


def test_user_correction_overrides_same_pattern(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add("SQDC", "Achats", source="legacy")
    assert rules.add("SQDC", "Santé", source="user") is True
    assert rules.match("SQDC MONTREAL") == "Santé"
    assert len(rules) == 1


def test_duplicate_pattern_not_added(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    assert rules.add("METRO", "Épicerie") is True
    assert rules.add("metro", "Épicerie") is False
    assert len(rules) == 1


def test_empty_pattern_rejected(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    assert rules.add("   ", "Autre") is False
    assert len(rules) == 0


def test_regex_rule(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add(r"^UBER\b", "Transport", regex=True)
    assert rules.match("UBER TRIP") == "Transport"
    assert rules.match("KLUBER SHOP") is None


def test_invalid_regex_is_skipped(tmp_path: Path) -> None:
    rules = make_store(tmp_path)
    rules.add("[invalid", "Autre", regex=True)
    assert rules.match("ANYTHING") is None


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    first = RuleStore(path=path)
    first.add("METRO", "Épicerie", source="legacy")
    first.save()

    second = RuleStore(path=path)
    assert second.match("METRO RICHELIEU") == "Épicerie"
    rule = second.rules[0]
    assert rule["source"] == "legacy"
    assert rule["created"]


def test_corrupt_file_recovers(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    path.write_text("[broken", encoding="utf-8")
    rules = RuleStore(path=path)
    assert len(rules) == 0
