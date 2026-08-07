"""Privacy guard: the single place where outbound API payloads are built.

Only normalized merchant keys (plain strings) may be sent to the API for
categorization — never amounts, balances, transaction dates or account
identifiers. The whole AI layer must build its outbound content here.
"""

import json
from collections.abc import Sequence


def build_categorization_payload(merchant_keys: Sequence[str]) -> str:
    """Build the user message sent to the categorization API.

    Accepts only a sequence of strings; anything else (number, dict,
    DataFrame row…) raises TypeError. Amounts therefore cannot reach the
    API by construction.
    """
    if isinstance(merchant_keys, (str, bytes)):
        raise TypeError("merchant_keys must be a sequence of strings, not a single string")

    keys = list(merchant_keys)
    for key in keys:
        if not isinstance(key, str):
            raise TypeError(
                f"Only merchant strings may be sent to the API (got {type(key).__name__})"
            )
    return json.dumps(keys, ensure_ascii=False)


_ALLOWED_SCALARS = (str, int, float, bool, type(None))


def build_narrative_payload(summary: dict) -> str:
    """Build the user message for the monthly narrative: aggregates only.

    Accepts plain JSON-serializable structures (dicts, lists, strings and
    numbers). Any other object — a DataFrame, a Series, raw transaction
    rows — raises TypeError, so transaction-level data cannot reach the API.
    """
    if not isinstance(summary, dict):
        raise TypeError(f"summary must be a dict (got {type(summary).__name__})")

    def _check(node, path: str) -> None:
        if isinstance(node, _ALLOWED_SCALARS):
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                _check(item, f"{path}[{i}]")
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path}: dict keys must be strings")
                _check(value, f"{path}.{key}")
            return
        raise TypeError(
            f"{path}: only plain JSON values may be sent to the API (got {type(node).__name__})"
        )

    _check(summary, "summary")
    return json.dumps(summary, ensure_ascii=False)


def build_query_payload(question: str) -> str:
    """Build the user message for query translation: the user's own question only.

    The question is user-authored text and may be sent verbatim; anything
    that is not a plain string raises TypeError.
    """
    if not isinstance(question, str):
        raise TypeError(
            f"Only the user's question string may be sent to the API (got {type(question).__name__})"
        )
    return question.strip()
