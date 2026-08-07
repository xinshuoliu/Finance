"""API call logging into data/api_log.jsonl (tokens, latency, cost).

Every Anthropic API call must go through logged_call() — this is what makes
the system's cost measurable (evaluation phase).
"""

import json
import time
from datetime import datetime, timezone

from ai.config import API_LOG_FILE, PRICING_USD_PER_MTOK


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Compute a call's cost from published pricing; None for unknown models."""
    pricing = PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        return None
    cost = input_tokens * pricing["input"] / 1_000_000 + output_tokens * pricing["output"] / 1_000_000
    return round(cost, 6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(record: dict) -> None:
    API_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(API_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(purpose: str, **fields) -> None:
    """Log a local event without an API call (e.g. a cascade run summary)."""
    _append({"ts": _now(), "purpose": purpose, **fields})


def logged_call(client, *, purpose: str, **create_kwargs):
    """Call client.messages.create while logging model, tokens, latency and cost."""
    model = create_kwargs.get("model", "")
    start = time.perf_counter()
    try:
        response = client.messages.create(**create_kwargs)
    except Exception as exc:
        _append(
            {
                "ts": _now(),
                "purpose": purpose,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    _append(
        {
            "ts": _now(),
            "purpose": purpose,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "latency_ms": latency_ms,
            "cost_usd": compute_cost_usd(model, input_tokens, output_tokens),
        }
    )
    return response
