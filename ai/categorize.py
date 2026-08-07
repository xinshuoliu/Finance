"""Cascading categorization: cache -> rules -> LLM (Claude Haiku).

Tiers, cheapest first:
1. Merchant cache — zero cost, deterministic (same key, same category);
2. User/legacy rules — deterministic;
3. Batched LLM calls (~30 merchants/request), for unknown merchants only.

On API failure, unknown merchants get "Autre" with the needs-review flag and
no cache write — they are retried later.

Note: the categorizer prompt is intentionally French — it matches the French
category set and the mostly-Quebecois merchant strings.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass

import anthropic

from ai.cache import MerchantCache
from ai.config import (
    CATEGORIES,
    CATEGORIZER_BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    MODEL_CATEGORIZER,
    ai_available,
)
from ai.llm_log import log_event, logged_call
from ai.privacy import build_categorization_payload
from ai.rules import RuleStore


@dataclass
class Categorization:
    """Categorization result for one merchant key."""

    category: str
    confidence: float
    source: str  # "cache" | "rule" | "llm" | "fallback"
    needs_review: bool = False


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic | None:
    global _client
    if not ai_available():
        return None
    if _client is None:
        _client = anthropic.Anthropic(timeout=30.0)
    return _client


# Structured-outputs schema: the category field is locked to the closed set,
# so the model cannot invent categories. Code-side validation stays anyway.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "merchant": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "confidence": {"type": "number"},
                },
                "required": ["merchant", "category", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Tu classes des libellés de marchands provenant de relevés bancaires canadiens (souvent québécois) dans une liste fermée de catégories.

Catégories permises :
- Épicerie : supermarchés et marchés d'alimentation (Metro, IGA, Provigo, Costco…)
- Restaurant : restaurants, cafés, restauration rapide, bars
- Achats : magasins et achats en ligne (vêtements, électronique, Amazon, SQDC…)
- Transport : essence, transport en commun, stationnement, taxi, recharge OPUS
- Logement : loyer, hypothèque, assurance habitation, rénovation
- Services publics : électricité, internet, téléphone (Hydro-Québec, Bell, Vidéotron…)
- Santé : pharmacies, cliniques, dentistes, optométristes
- Loisirs : sport, plein air, cinéma, voyages, hôtels, activités
- Abonnements : services numériques récurrents (Netflix, Spotify, logiciels)
- Revenu : salaires, dépôts, remboursements reçus
- Transfert : virements entre comptes, paiements de carte de crédit
- Autre : ce qui ne correspond à aucune catégorie

Réponds pour CHAQUE libellé fourni, en recopiant le libellé à l'identique dans le champ « merchant ». Le champ « confidence » est ta probabilité (entre 0 et 1) que la catégorie choisie soit la bonne ; sois honnête et utilise une valeur basse si le marchand est ambigu ou inconnu."""


def _parse_response(text: str, batch: list[str]) -> tuple[dict[str, tuple[str, float]] | None, bool]:
    """Extract {merchant: (category, confidence)} from a model response.

    Returns (None, True) when the response is unreadable; otherwise the dict
    of valid items plus a flag signalling invalid items that were skipped.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, True
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None, True

    wanted = set(batch)
    out: dict[str, tuple[str, float]] = {}
    had_invalid = False
    for item in items:
        if not isinstance(item, dict):
            had_invalid = True
            continue
        merchant = item.get("merchant")
        category = item.get("category")
        if not isinstance(merchant, str) or merchant not in wanted or category not in CATEGORIES:
            had_invalid = True
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        out.setdefault(merchant, (category, min(max(confidence, 0.0), 1.0)))
    return out, had_invalid


def _attempt_batch(client, batch: list[str]) -> tuple[dict[str, tuple[str, float]] | None, bool]:
    response = logged_call(
        client,
        purpose="categorize",
        model=MODEL_CATEGORIZER,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_categorization_payload(batch)}],
        output_config={"format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
    )
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    return _parse_response(text, batch)


def _categorize_batch(client, batch: list[str]) -> dict[str, Categorization]:
    """Categorize one batch via the API, retrying once on invalid output."""
    first, had_invalid = _attempt_batch(client, batch)
    merged = dict(first or {})
    if first is None or had_invalid or any(key not in merged for key in batch):
        second, _ = _attempt_batch(client, batch)
        for key, value in (second or {}).items():
            merged.setdefault(key, value)

    out: dict[str, Categorization] = {}
    for key in batch:
        if key in merged:
            category, confidence = merged[key]
            out[key] = Categorization(category, confidence, "llm", confidence < CONFIDENCE_THRESHOLD)
        else:
            # The model silently omitted (or returned invalid output for) this merchant
            out[key] = Categorization("Autre", 0.0, "llm", True)
    return out


def categorize_keys(
    keys: Iterable[str],
    cache: MerchantCache,
    rules: RuleStore,
    client=None,
    use_llm: bool = True,
) -> dict[str, Categorization]:
    """Categorize merchant keys through the cache -> rules -> LLM cascade.

    LLM verdicts are written to the cache (flagged needs_review under the
    confidence threshold). API transport failures produce uncached
    "fallback" results, so they retry on a later run.
    """
    results: dict[str, Categorization] = {}
    unknown: list[str] = []

    for key in dict.fromkeys(k for k in keys if k):
        entry = cache.get(key)
        if entry is not None:
            results[key] = Categorization(
                category=entry.get("category", "Autre"),
                confidence=float(entry.get("confidence", 0.0)),
                source="cache",
                needs_review=bool(entry.get("needs_review", False)),
            )
            continue
        rule_category = rules.match(key)
        if rule_category is not None:
            results[key] = Categorization(rule_category, 1.0, "rule")
            continue
        unknown.append(key)

    llm_count = fallback_count = 0
    if unknown:
        if use_llm:
            client = client if client is not None else _get_client()
        else:
            client = None

        if client is None:
            for key in unknown:
                results[key] = Categorization("Autre", 0.0, "fallback", True)
            fallback_count = len(unknown)
        else:
            cached_any = False
            for i in range(0, len(unknown), CATEGORIZER_BATCH_SIZE):
                batch = unknown[i : i + CATEGORIZER_BATCH_SIZE]
                try:
                    batch_results = _categorize_batch(client, batch)
                except Exception:
                    # Transport/API failure: don't cache, so these retry next run
                    for key in batch:
                        results[key] = Categorization("Autre", 0.0, "fallback", True)
                    fallback_count += len(batch)
                    continue
                for key, res in batch_results.items():
                    results[key] = res
                    cache.set(key, res.category, res.confidence, "llm", needs_review=res.needs_review)
                    cached_any = True
                llm_count += len(batch_results)
            if cached_any:
                cache.save()

        log_event(
            "categorize_run",
            total_keys=len(results),
            cache_hits=sum(1 for r in results.values() if r.source == "cache"),
            rule_hits=sum(1 for r in results.values() if r.source == "rule"),
            llm_categorized=llm_count,
            fallback=fallback_count,
        )

    return results


def record_user_correction(
    key: str,
    category: str,
    cache: MerchantCache,
    rules: RuleStore,
    save: bool = True,
) -> bool:
    """Record a user's category decision as a durable rule + cache entry.

    The rule makes the correction apply to every past and future transaction
    of this merchant (and survives re-importing the same statement); the
    cache entry clears the needs-review flag with full confidence. With
    save=False the caller batches several corrections and saves once.
    """
    if not key:
        return False
    rules.add(key, category, source="user")
    cache.set(key, category, 1.0, "user")
    if save:
        rules.save()
        cache.save()
    return True
