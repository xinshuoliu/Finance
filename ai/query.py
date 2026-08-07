"""Natural-language questions translated into validated filter specs.

The model never returns code — only a JSON filter object, schema-locked via
structured outputs and re-validated with Pydantic. The spec is then executed
locally against the DataFrame by our own pandas code (execute_spec).
"""

import json
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from ai.client import get_client
from ai.config import CATEGORIES, MODEL_QUERY
from ai.llm_log import logged_call
from ai.privacy import build_query_payload


class QueryTranslationError(Exception):
    """User-facing error: the question could not become a valid filter spec."""


class FilterSpec(BaseModel):
    """Validated filter specification — the only thing the model may return."""

    model_config = ConfigDict(extra="forbid")

    categories: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None
    aggregate: Literal["sum", "mean", "count", "max"] = "sum"
    group_by: Literal["month", "week", "category", "merchant"] | None = None
    merchant_contains: str | None = None
    transaction_type: Literal["debit", "credit"] | None = "debit"

    @field_validator("categories")
    @classmethod
    def _categories_in_closed_set(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unknown = [c for c in value if c not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories: {', '.join(unknown)}")
        return value or None

    @field_validator("merchant_contains")
    @classmethod
    def _strip_merchant(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _dates_in_order(self) -> "FilterSpec":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        return self


@dataclass
class QueryResult:
    """Result of executing a FilterSpec against the transactions DataFrame."""

    value: float | int | None  # scalar result when group_by is None
    table: pd.DataFrame | None  # grouped result otherwise
    filtered: pd.DataFrame  # the matching transactions
    description: str  # human-readable interpretation of the spec


_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "anyOf": [
                {"type": "array", "items": {"type": "string", "enum": CATEGORIES}},
                {"type": "null"},
            ]
        },
        "date_from": {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]},
        "date_to": {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]},
        "aggregate": {"type": "string", "enum": ["sum", "mean", "count", "max"]},
        "group_by": {
            "anyOf": [
                {"type": "string", "enum": ["month", "week", "category", "merchant"]},
                {"type": "null"},
            ]
        },
        "merchant_contains": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "transaction_type": {
            "anyOf": [{"type": "string", "enum": ["debit", "credit"]}, {"type": "null"}]
        },
    },
    "required": [
        "categories",
        "date_from",
        "date_to",
        "aggregate",
        "group_by",
        "merchant_contains",
        "transaction_type",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT_TEMPLATE = """You translate a user's question about their personal bank transactions into a JSON filter specification. You never write code, SQL or prose — only the filter object. The filter is executed locally by the application.

Rules:
- categories: pick from the allowed list only, or null for all categories.
- date_from / date_to: ISO dates (YYYY-MM-DD) or null. Resolve relative expressions ("since January", "last month") against today's date. When the user names a month without a year, use the most recent occurrence.
- aggregate: "sum" (total spent), "mean" (average), "count" (number of transactions), "max" (largest single transaction).
- group_by: "month", "week", "category" or "merchant" when the user wants a breakdown or a trend, else null.
- merchant_contains: a short substring of the merchant name when the user asks about a specific merchant, else null.
- transaction_type: "debit" for spending questions, "credit" for income or money received, null for both.

Allowed categories: {categories}
Questions may be in English or French; category names are French.
Today's date: {today}"""


def translate_question(question: str, client=None, today: date | None = None) -> FilterSpec:
    """Translate a natural-language question into a validated FilterSpec.

    Raises QueryTranslationError with a user-facing message when the AI is
    unavailable or returns anything that fails validation.
    """
    question = str(question).strip()
    if not question:
        raise QueryTranslationError("Please type a question first.")

    if client is None:
        client = get_client()
    if client is None:
        raise QueryTranslationError("AI is not configured — add ANTHROPIC_API_KEY to a .env file.")

    system = SYSTEM_PROMPT_TEMPLATE.format(
        categories=", ".join(CATEGORIES), today=(today or date.today()).isoformat()
    )
    try:
        response = logged_call(
            client,
            purpose="query",
            model=MODEL_QUERY,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": build_query_payload(question)}],
            output_config={"format": {"type": "json_schema", "schema": _QUERY_SCHEMA}},
        )
    except Exception as exc:
        raise QueryTranslationError("The AI request failed — please try again.") from exc

    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueryTranslationError(
            "The AI returned an unreadable answer — please rephrase your question."
        ) from exc

    try:
        return FilterSpec.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise QueryTranslationError(
            f"The AI returned an invalid filter ({first.get('msg', 'validation error')}) — "
            "please rephrase your question."
        ) from exc


_AGG_VALUE_LABEL = {"sum": "Total", "mean": "Average", "count": "Count", "max": "Maximum"}
_AGG_TEXT = {"sum": "Total", "mean": "Average", "count": "Number", "max": "Largest amount"}
_TYPE_TEXT = {"debit": "expenses", "credit": "incoming payments", None: "transactions"}


def describe_spec(spec: FilterSpec) -> str:
    """Human-readable interpretation of a spec, shown above the result."""
    parts = [f"{_AGG_TEXT[spec.aggregate]} of {_TYPE_TEXT[spec.transaction_type]}"]
    if spec.categories:
        parts.append("categories: " + ", ".join(spec.categories))
    if spec.date_from:
        parts.append(f"from {spec.date_from.isoformat()}")
    if spec.date_to:
        parts.append(f"to {spec.date_to.isoformat()}")
    if spec.merchant_contains:
        parts.append(f'merchant contains "{spec.merchant_contains}"')
    if spec.group_by:
        parts.append(f"grouped by {spec.group_by}")
    return " • ".join(parts)


def execute_spec(spec: FilterSpec, df: pd.DataFrame) -> QueryResult:
    """Execute a validated spec against the transactions DataFrame (pandas only)."""
    filtered = df.copy()

    if spec.transaction_type == "debit":
        filtered = filtered[filtered["Debit/Credit"] == "Debit"]
    elif spec.transaction_type == "credit":
        filtered = filtered[filtered["Debit/Credit"] == "Credit"]

    if spec.categories:
        filtered = filtered[filtered["Category"].isin(spec.categories)]
    if spec.date_from is not None:
        filtered = filtered[filtered["Date"].dt.date >= spec.date_from]
    if spec.date_to is not None:
        filtered = filtered[filtered["Date"].dt.date <= spec.date_to]
    if spec.merchant_contains:
        needle = spec.merchant_contains.lower()
        mask = filtered["Details"].astype(str).str.lower().str.contains(needle, regex=False)
        if "Merchant" in filtered.columns:
            mask |= filtered["Merchant"].astype(str).str.lower().str.contains(needle, regex=False)
        filtered = filtered[mask]

    description = describe_spec(spec)

    if spec.group_by is None:
        amounts = filtered["Amount"]
        if spec.aggregate == "count":
            value: float | int = int(len(amounts))
        elif len(amounts) == 0:
            value = 0.0
        elif spec.aggregate == "sum":
            value = float(amounts.sum())
        elif spec.aggregate == "mean":
            value = float(amounts.mean())
        else:
            value = float(amounts.max())
        return QueryResult(value=value, table=None, filtered=filtered, description=description)

    if spec.group_by == "month":
        groups, label = filtered["Date"].dt.to_period("M").astype(str), "Month"
    elif spec.group_by == "week":
        groups, label = filtered["Date"].dt.to_period("W").astype(str), "Week"
    elif spec.group_by == "category":
        groups, label = filtered["Category"], "Category"
    else:
        merchant_col = filtered["Merchant"] if "Merchant" in filtered.columns else filtered["Details"]
        groups, label = merchant_col, "Merchant"

    grouped = filtered.groupby(groups)["Amount"]
    if spec.aggregate == "sum":
        series = grouped.sum()
    elif spec.aggregate == "mean":
        series = grouped.mean()
    elif spec.aggregate == "count":
        series = grouped.size()
    else:
        series = grouped.max()

    table = series.reset_index()
    table.columns = [label, _AGG_VALUE_LABEL[spec.aggregate]]
    if spec.group_by in ("category", "merchant"):
        table = table.sort_values(table.columns[1], ascending=False)
    else:
        table = table.sort_values(label)
    return QueryResult(
        value=None, table=table.reset_index(drop=True), filtered=filtered, description=description
    )
