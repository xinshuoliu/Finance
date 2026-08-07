"""Fixtures communes : isole le journal API dans un répertoire temporaire."""

import pytest

import ai.llm_log


@pytest.fixture(autouse=True)
def isolated_api_log(tmp_path, monkeypatch):
    monkeypatch.setattr(ai.llm_log, "API_LOG_FILE", tmp_path / "api_log.jsonl")
