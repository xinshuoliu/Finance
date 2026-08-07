"""Cache persistant des catégorisations de marchands.

Associe chaque clé marchande normalisée (voir ai.normalize) à sa catégorie.
Un marchand déjà catégorisé ne repasse jamais par l'API : même entrée ->
même catégorie, à coût nul et de façon déterministe.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ai.config import MERCHANT_CACHE_FILE


class MerchantCache:
    """Cache disque « clé marchande -> {category, confidence, source, updated_at} »."""

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
        """Retourne l'entrée du cache pour une clé marchande, ou None si absente."""
        return self._data.get(key)

    def set(
        self,
        key: str,
        category: str,
        confidence: float,
        source: str,
        needs_review: bool = False,
    ) -> None:
        """Enregistre en mémoire la catégorie d'une clé marchande.

        Appeler save() ensuite pour écrire sur disque — cela permet de
        grouper l'écriture après un lot de catégorisations.
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
        """Écrit le cache sur disque de façon atomique (fichier temporaire puis rename)."""
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
