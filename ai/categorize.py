"""Catégorisation en cascade : cache -> règles -> LLM (Claude Haiku).

Ordre des niveaux, du moins cher au plus cher :
1. Cache marchand — coût nul, déterministe (même clé, même catégorie) ;
2. Règles utilisateur/héritées — déterministe ;
3. Appel LLM par lots (~30 marchands/requête), pour les inconnus seulement.

En cas d'échec de l'API, les inconnus reçoivent « Autre » avec le drapeau
« à revoir », sans écriture au cache — ils seront réessayés plus tard.
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
    """Résultat de catégorisation d'une clé marchande."""

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
    """Extrait {marchand: (catégorie, confiance)} d'une réponse du modèle.

    Retourne (None, True) si la réponse est illisible ; sinon le dict des
    éléments valides et un booléen signalant des éléments invalides ignorés.
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
    """Catégorise un lot via l'API, avec une seule relance si la sortie est invalide."""
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
    """Catégorise des clés marchandes via la cascade cache -> règles -> LLM.

    Les résultats LLM sont écrits au cache (avec « needs_review » sous le
    seuil de confiance). Un échec de transport API produit des résultats
    « fallback » non mis en cache, pour réessai ultérieur.
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
