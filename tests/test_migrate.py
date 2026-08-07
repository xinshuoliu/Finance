"""Tests de la migration de l'ancien système de catégories."""

import json
from pathlib import Path

from ai.migrate import map_legacy_category, migrate_legacy
from ai.rules import RuleStore

# Shape copied from this project's real categories.json
LEGACY_CATEGORIES = {
    "Uncategorized": [],
    "Shopping": ["HUGO BOSS - 6186 St-Bruno-de-MQC", "SQDC77001 SQDC.CA MONTREAL QC"],
    "restaurant": ["MCDONALD'S #22028 MONT-TREMBLANQC"],
    "gym": ["ROSE BLOC GREENFIELD PAQC"],
    "Activities": ["SKI MONT BLANC MONT-BLANC QC"],
    "transport": ["CHRONO-RECHARGE OPUS MONTREAL QC"],
    "insurance": ["BOOKING.COM"],
}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_migration_creates_rules_and_rekeys_budgets(tmp_path: Path) -> None:
    categories = tmp_path / "categories.json"
    budgets = tmp_path / "budgets.json"
    write_json(categories, LEGACY_CATEGORIES)
    write_json(budgets, {"restaurant": 200.0, "Activities": 100.0, "gym": 100.0})

    rules = RuleStore(path=tmp_path / "rules.json")
    summary = migrate_legacy(str(categories), str(budgets), rules)

    assert summary["rules_added"] == 7
    assert rules.match("MCDONALD'S") == "Restaurant"
    assert rules.match("HUGO BOSS") == "Achats"
    assert rules.match("SKI MONT BLANC") == "Loisirs"
    assert rules.match("BOOKING.COM") == "Loisirs"
    assert rules.match("CHRONO-RECHARGE OPUS") == "Transport"

    migrated = json.loads(budgets.read_text(encoding="utf-8"))
    assert migrated == {"Restaurant": 200.0, "Loisirs": 200.0}

    # Rules file written even with nothing more to add: marks migration as done
    assert (tmp_path / "rules.json").exists()


def test_migration_idempotent(tmp_path: Path) -> None:
    categories = tmp_path / "categories.json"
    budgets = tmp_path / "budgets.json"
    write_json(categories, LEGACY_CATEGORIES)
    write_json(budgets, {"gym": 100.0})

    rules = RuleStore(path=tmp_path / "rules.json")
    first = migrate_legacy(str(categories), str(budgets), rules)
    second = migrate_legacy(str(categories), str(budgets), rules)

    assert first["rules_added"] == 7
    assert second["rules_added"] == 0
    assert len(rules) == 7
    # Budgets already re-keyed on the first pass; second pass leaves them alone
    assert json.loads(budgets.read_text(encoding="utf-8")) == {"Loisirs": 100.0}


def test_map_legacy_category() -> None:
    assert map_legacy_category("Shopping") == "Achats"
    assert map_legacy_category("weird stuff") == "Autre"
    assert map_legacy_category("Uncategorized") is None
    assert map_legacy_category("Transport") == "Transport"  # already-canonical passes through


def test_missing_files_are_fine(tmp_path: Path) -> None:
    rules = RuleStore(path=tmp_path / "rules.json")
    summary = migrate_legacy(str(tmp_path / "none.json"), str(tmp_path / "none2.json"), rules)
    assert summary == {"rules_added": 0, "budgets_migrated": 0}
    assert (tmp_path / "rules.json").exists()


def test_canonical_budgets_untouched(tmp_path: Path) -> None:
    budgets = tmp_path / "budgets.json"
    write_json(budgets, {"Restaurant": 150.0})
    rules = RuleStore(path=tmp_path / "rules.json")
    migrate_legacy(str(tmp_path / "missing.json"), str(budgets), rules)
    assert json.loads(budgets.read_text(encoding="utf-8")) == {"Restaurant": 150.0}
