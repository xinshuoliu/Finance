"""AI layer configuration: models, pricing, paths and thresholds."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
MERCHANT_CACHE_FILE = DATA_DIR / "merchant_cache.json"
CATEGORY_RULES_FILE = DATA_DIR / "category_rules.json"
API_LOG_FILE = DATA_DIR / "api_log.jsonl"

# Closed category set — the model may never invent categories outside this list.
# "Achats" was added to the original 11: the existing data clearly needs a
# general-shopping bucket that the spec's list did not have.
CATEGORIES = [
    "Épicerie",
    "Restaurant",
    "Achats",
    "Transport",
    "Logement",
    "Services publics",
    "Santé",
    "Loisirs",
    "Abonnements",
    "Revenu",
    "Transfert",
    "Autre",
]

MODEL_CATEGORIZER = "claude-haiku-4-5-20251001"
MODEL_QUERY = "claude-haiku-4-5-20251001"
MODEL_NARRATIVE = "claude-sonnet-5"

# =====================================================================
# PRICING — USD per 1 MILLION tokens. Update here when Anthropic's
# published pricing changes (https://platform.claude.com/docs/en/pricing).
# Verified 2026-08-07. claude-sonnet-5 has intro pricing ($2/$10) through
# 2026-08-31; we use the standard list price for conservative estimates.
# =====================================================================
PRICING_USD_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}

CATEGORIZER_BATCH_SIZE = 30
CONFIDENCE_THRESHOLD = 0.7


def ai_available() -> bool:
    """Return True when the Anthropic API key is configured (AI features active)."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))
