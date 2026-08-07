"""Tests for API call logging (cost, latency, errors)."""

import json
from types import SimpleNamespace

import pytest

import ai.llm_log as llm_log


def make_client(response=None, error=None):
    def create(**kwargs):
        if error is not None:
            raise error
        return response

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def read_log() -> list[dict]:
    lines = llm_log.API_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_compute_cost_known_model() -> None:
    assert llm_log.compute_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0


def test_compute_cost_unknown_model() -> None:
    assert llm_log.compute_cost_usd("mystery-model", 1000, 1000) is None


def test_logged_call_writes_record() -> None:
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=500, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    client = make_client(response=SimpleNamespace(content=[], usage=usage))
    llm_log.logged_call(
        client, purpose="test", model="claude-haiku-4-5-20251001", max_tokens=10, messages=[]
    )
    (record,) = read_log()
    assert record["purpose"] == "test"
    assert record["model"] == "claude-haiku-4-5-20251001"
    assert record["input_tokens"] == 1000
    assert record["output_tokens"] == 500
    assert record["latency_ms"] >= 0
    assert record["cost_usd"] == pytest.approx(0.0035)


def test_logged_call_logs_and_reraises_errors() -> None:
    client = make_client(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        llm_log.logged_call(client, purpose="test", model="m", messages=[])
    (record,) = read_log()
    assert "boom" in record["error"]


def test_log_event() -> None:
    llm_log.log_event("categorize_run", cache_hits=5)
    (record,) = read_log()
    assert record["purpose"] == "categorize_run"
    assert record["cache_hits"] == 5
