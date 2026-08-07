"""Monthly narrative: pandas computes the figures, the model only writes prose.

The model receives a pre-computed summary dict and must not introduce any
number that is not in it. After generation, every numeric token in the
response is checked against the summary; the ones missing are flagged and
logged — that is the hallucination signal the spec asks for.
"""

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from ai.client import get_client
from ai.config import MODEL_NARRATIVE
from ai.llm_log import log_event, logged_call
from ai.privacy import build_narrative_payload


class NarrativeError(Exception):
    """User-facing error: the narrative could not be generated."""


@dataclass
class NarrativeResult:
    """Generated narrative plus the numeric tokens that failed grounding."""

    text: str
    suspect_numbers: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You write a short monthly summary for a personal finance dashboard.

You receive pre-computed figures as JSON: spending by category, percentage changes versus the previous month, new merchants, detected recurring subscriptions and budget status. Amounts are in CAD.

Write 3 to 5 plain-language sentences in English observing the notable points: biggest spending categories, significant changes, new merchants, subscriptions, budgets exceeded or respected.

STRICT RULES:
- Never introduce any number that is not present in the input JSON. No arithmetic of your own, no estimates, no re-rounding. When you mention a number, copy it exactly from the input.
- Only state facts present in the input.
- Plain prose only: no headings, no bullet points, no markdown."""


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def allowed_numbers(summary: dict) -> set[float]:
    """Every number the narrative may legitimately mention.

    All numeric tokens appearing anywhere in the summary (including inside
    strings like "2026-01"), plus the length of each list — "3 subscriptions"
    is grounded when recurring_detected has 3 entries.
    """
    allowed: set[float] = set()
    for token in _extract_numbers(json.dumps(summary, ensure_ascii=False)):
        value = _to_float(token)
        if value is not None:
            allowed.add(value)

    def _walk(node) -> None:
        if isinstance(node, list):
            allowed.add(float(len(node)))
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            for item in node.values():
                _walk(item)

    _walk(summary)
    return allowed


def check_numbers(text: str, summary: dict) -> list[str]:
    """Return the numeric tokens of the text that are absent from the summary."""
    allowed = allowed_numbers(summary)
    suspects: list[str] = []
    for token in _extract_numbers(text):
        value = _to_float(token)
        if value is None:
            continue
        if not any(abs(value - candidate) <= 0.005 for candidate in allowed):
            suspects.append(token)
    return list(dict.fromkeys(suspects))


def build_month_summary(
    df: pd.DataFrame,
    month: str,
    budgets: dict | None = None,
    recurring: pd.DataFrame | None = None,
) -> dict:
    """Compute one month's figures — all the math happens here, in pandas."""
    debits = df[df["Debit/Credit"] == "Debit"]
    debit_periods = debits["Date"].dt.to_period("M").astype(str)
    current = debits[debit_periods == month]
    previous = debits[debit_periods == str(pd.Period(month) - 1)]

    by_category = (
        current.groupby("Category")["Amount"].sum().round(2).sort_values(ascending=False)
    )

    vs_prev: dict[str, float] = {}
    prev_by_cat = previous.groupby("Category")["Amount"].sum()
    for category, amount in by_category.items():
        prev_amount = float(prev_by_cat.get(category, 0.0))
        if prev_amount > 0:
            vs_prev[category] = round((float(amount) - prev_amount) / prev_amount * 100, 1)

    # "New" merchants only mean something once there is history before this month
    new_merchants: list[dict] = []
    if "Merchant" in debits.columns and bool((debit_periods < month).any()):
        first_seen = debits.groupby("Merchant")["Date"].min().dt.to_period("M").astype(str)
        new_keys = set(first_seen[first_seen == month].index) - {""}
        spend = (
            current[current["Merchant"].isin(new_keys)]
            .groupby("Merchant")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        new_merchants = [
            {"merchant": merchant, "amount": round(float(amount), 2)}
            for merchant, amount in spend.items()
        ]

    recurring_detected: list[dict] = []
    if recurring is not None and not recurring.empty:
        recurring_detected = [
            {
                "merchant": row.Merchant,
                "frequency": row.Frequency,
                "average_amount": float(row.AverageAmount),
            }
            for row in recurring.head(10).itertuples()
        ]

    budget_status: dict[str, dict] = {}
    for category, budget in (budgets or {}).items():
        budget_value = float(budget or 0)
        if budget_value > 0:
            budget_status[category] = {
                "budget": round(budget_value, 2),
                "spent": round(float(by_category.get(category, 0.0)), 2),
            }

    return {
        "month": month,
        "total_spent": round(float(current["Amount"].sum()), 2),
        "transaction_count": int(len(current)),
        "by_category": {category: float(amount) for category, amount in by_category.items()},
        "vs_prev_month_pct": vs_prev,
        "top_new_merchants": new_merchants,
        "recurring_detected": recurring_detected,
        "budget_status": budget_status,
    }


def generate_narrative(summary: dict, client=None) -> NarrativeResult:
    """Generate the monthly narrative from pre-computed figures.

    Raises NarrativeError with a user-facing message when the AI is
    unavailable or the request fails.
    """
    if client is None:
        client = get_client()
    if client is None:
        raise NarrativeError("AI is not configured — add ANTHROPIC_API_KEY to a .env file.")

    try:
        response = logged_call(
            client,
            purpose="narrative",
            model=MODEL_NARRATIVE,
            max_tokens=1000,
            thinking={"type": "disabled"},  # short grounded prose needs no reasoning budget
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_narrative_payload(summary)}],
        )
    except Exception as exc:
        raise NarrativeError("The AI request failed — please try again.") from exc

    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise NarrativeError("The AI returned an empty answer — please try again.")

    suspects = check_numbers(text, summary)
    log_event(
        "narrative_check",
        month=summary.get("month"),
        numbers_total=len(_extract_numbers(text)),
        suspect_numbers=suspects,
    )
    return NarrativeResult(text=text, suspect_numbers=suspects)
