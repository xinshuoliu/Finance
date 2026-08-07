"""Migration ponctuelle de l'ancien système de catégories vers la couche IA.

Convertit les mots-clés appris (categories.json) en règles déterministes
(data/category_rules.json) et re-clé les budgets vers la liste fermée de
catégories françaises. S'exécute une seule fois : l'existence du fichier de
règles marque la migration comme faite.
"""

import json
import os

from ai.config import CATEGORIES
from ai.normalize import normalize
from ai.rules import RuleStore

# Legacy free-form category names found in this project's data -> closed set
LEGACY_CATEGORY_MAP = {
    "shopping": "Achats",
    "insurance": "Loisirs",  # its only learned keyword was BOOKING.COM (travel)
    "restaurant": "Restaurant",
    "activities": "Loisirs",
    "gym": "Loisirs",
    "transport": "Transport",
}


def map_legacy_category(name: str) -> str | None:
    """Traduit un ancien nom de catégorie vers la liste fermée (None = ignorer)."""
    if name in CATEGORIES:
        return name
    lowered = name.strip().lower()
    if lowered == "uncategorized":
        return None
    return LEGACY_CATEGORY_MAP.get(lowered, "Autre")


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def migrate_legacy(categories_path: str, budgets_path: str, rules: RuleStore) -> dict:
    """Migre mots-clés et budgets hérités ; retourne {rules_added, budgets_migrated}."""
    rules_added = 0
    legacy_categories = _read_json(categories_path)
    if isinstance(legacy_categories, dict):
        for legacy_name, keywords in legacy_categories.items():
            target = map_legacy_category(str(legacy_name))
            if target is None or not isinstance(keywords, list):
                continue
            for keyword in keywords:
                pattern = normalize(str(keyword))
                if pattern and rules.add(pattern, target, source="legacy"):
                    rules_added += 1
    # Always write the rules file: its existence marks the migration as done
    rules.save()

    budgets_migrated = 0
    budgets = _read_json(budgets_path)
    if isinstance(budgets, dict) and any(k not in CATEGORIES for k in budgets):
        migrated: dict[str, float] = {}
        for name, amount in budgets.items():
            target = map_legacy_category(str(name)) or "Autre"
            try:
                migrated[target] = migrated.get(target, 0.0) + float(amount)
            except (TypeError, ValueError):
                continue
            budgets_migrated += 1
        with open(budgets_path, "w", encoding="utf-8") as f:
            json.dump(migrated, f, ensure_ascii=False)

    return {"rules_added": rules_added, "budgets_migrated": budgets_migrated}
