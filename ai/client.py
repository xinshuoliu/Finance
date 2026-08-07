"""Shared Anthropic client factory."""

import anthropic

from ai.config import ai_available

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic | None:
    """Return a shared Anthropic client, or None when no API key is configured."""
    global _client
    if not ai_available():
        return None
    if _client is None:
        _client = anthropic.Anthropic(timeout=30.0)
    return _client
