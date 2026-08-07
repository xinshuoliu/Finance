"""Persistent merchant categorization cache.

Maps each normalized merchant key (see ai.normalize) to its category, so an
already-categorized merchant never goes back through the API: same input ->
same category, deterministically and at zero cost.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ai.config import MERCHANT_CACHE_FILE


class MerchantCache:
    """Disk cache: merchant key -> {category, confidence, source, updated_at}."""

    def __init__(self, path: str | Path = MERCHANT_CACHE_FILE) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt cache is discarded rather than crashing the app
            return
        if isinstance(data, dict):
            self._data = data

    def get(self, key: str) -> dict | None:
        """Return the cache entry for a merchant key, or None when absent."""
        return self._data.get(key)

    def set(
        self,
        key: str,
        category: str,
        confidence: float,
        source: str,
        needs_review: bool = False,
    ) -> None:
        """Record a merchant key's category in memory.

        Call save() afterwards to write to disk — this allows batching the
        write after a whole categorization run.
        """
        entry: dict = {
            "category": category,
            "confidence": float(confidence),
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if needs_review:
            entry["needs_review"] = True
        self._data[key] = entry

    def save(self) -> None:
        """Write the cache to disk atomically (temp file then rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)
