"""Deterministic categorization rules (data/category_rules.json).

Each rule maps a pattern (substring, or regex when "regex": true) to a
category from the closed set. Rules apply to normalized merchant keys,
case-insensitively — same input, same category.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ai.config import CATEGORY_RULES_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RuleStore:
    """Ordered list of pattern -> category rules (first match wins)."""

    def __init__(self, path: str | Path = CATEGORY_RULES_FILE) -> None:
        self.path = Path(path)
        self._rules: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, list):
            self._rules = [r for r in data if isinstance(r, dict) and r.get("pattern")]

    @property
    def rules(self) -> list[dict]:
        """Copy of the rule list, in application order."""
        return list(self._rules)

    def add(self, pattern: str, category: str, source: str = "user", regex: bool = False) -> bool:
        """Add a rule; return True when something changed.

        An existing pattern is not duplicated, but a "user" rule replaces
        the category of an existing pattern (the user's decision wins).
        "user" rules are inserted first so they take precedence over
        inherited rules.
        """
        pattern = str(pattern).strip()
        if not pattern:
            return False

        for rule in self._rules:
            if rule["pattern"].lower() == pattern.lower() and bool(rule.get("regex")) == regex:
                if source == "user" and rule["category"] != category:
                    rule["category"] = category
                    rule["source"] = source
                    rule["created"] = _now()
                    return True
                return False

        rule = {"pattern": pattern, "category": category, "source": source, "created": _now()}
        if regex:
            rule["regex"] = True
        if source == "user":
            self._rules.insert(0, rule)
        else:
            self._rules.append(rule)
        return True

    def match(self, key: str) -> str | None:
        """Return the category of the first rule matching the key, or None."""
        if not key:
            return None
        lowered = key.lower()
        for rule in self._rules:
            if rule.get("regex"):
                try:
                    if re.search(rule["pattern"], key, re.IGNORECASE):
                        return rule["category"]
                except re.error:
                    continue
            elif rule["pattern"].lower() in lowered:
                return rule["category"]
        return None

    def save(self) -> None:
        """Write the rules to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._rules, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def __len__(self) -> int:
        return len(self._rules)
