"""Tests for the evaluation harness (offline: no real API calls)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eval.run_eval import (
    ConfigResult,
    build_template,
    confusion_matrix,
    load_labels,
    render_markdown,
    run_config,
    split_labels,
    update_readme,
)

STATEMENT = """Date,Description,Amount
2026-01-05,MCDONALD'S #22028 MONT-TREMBLANQC,20.00
2026-01-12,MCDONALD'S #9451 MONTREAL QC,15.00
2026-01-15,NETFLIX.COM 8665797172 ON,16.99
2026-02-15,NETFLIX.COM 8665797172 ON,16.99
2026-02-20,METRO 388 MONTREAL QC,80.00
"""


@pytest.fixture
def statement(tmp_path: Path) -> Path:
    path = tmp_path / "statement.csv"
    path.write_text(STATEMENT, encoding="utf-8")
    return path


def make_labels(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "merchant_key": key,
                "category": category,
                "occurrences": count,
                "example_description": key,
            }
            for key, category, count in rows
        ]
    )


# ---------- template ----------


def test_template_dedupes_and_counts(statement, tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    template = build_template([str(statement)], labels_path=labels_path)

    assert list(template.columns) == [
        "merchant_key",
        "category",
        "occurrences",
        "example_description",
    ]
    keys = dict(zip(template["merchant_key"], template["occurrences"]))
    # Both McDonald's rows collapse to one merchant, as do both Netflix rows
    assert keys == {"MCDONALD'S": 2, "NETFLIX.COM": 2, "METRO": 1}
    assert (template["category"] == "").all()
    assert labels_path.exists()


def test_template_preserves_existing_labels(statement, tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    build_template([str(statement)], labels_path=labels_path)

    saved = pd.read_csv(labels_path)
    saved["category"] = saved["category"].astype(object)
    saved.loc[saved["merchant_key"] == "NETFLIX.COM", "category"] = "Abonnements"
    saved.to_csv(labels_path, index=False)

    refreshed = build_template([str(statement)], labels_path=labels_path)
    labelled = dict(zip(refreshed["merchant_key"], refreshed["category"].fillna("")))
    assert labelled["NETFLIX.COM"] == "Abonnements"
    assert labelled["METRO"] == ""


# ---------- labels ----------


def test_load_labels_keeps_only_labelled_rows(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    make_labels([("A", "Restaurant", 3), ("B", "", 1)]).to_csv(path, index=False)
    labels = load_labels(path)
    assert list(labels["merchant_key"]) == ["A"]
    assert labels.iloc[0]["occurrences"] == 3


def test_load_labels_rejects_unknown_category(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    make_labels([("A", "Groceries", 1)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown categories in .*: Groceries"):
        load_labels(path)


def test_load_labels_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "nope.csv")


# ---------- split ----------


def test_split_is_deterministic_and_disjoint() -> None:
    labels = make_labels([(f"M{i}", "Autre", 1) for i in range(40)])
    train_a, test_a = split_labels(labels, seed=42)
    train_b, test_b = split_labels(labels, seed=42)

    assert list(train_a["merchant_key"]) == list(train_b["merchant_key"])
    assert list(test_a["merchant_key"]) == list(test_b["merchant_key"])
    assert not set(train_a["merchant_key"]) & set(test_a["merchant_key"])
    assert len(train_a) + len(test_a) == 40
    assert len(test_a) == 10  # 75/25 proportion when under the full 300/100


def test_split_uses_full_sizes_when_enough_labels() -> None:
    labels = make_labels([(f"M{i}", "Autre", 1) for i in range(500)])
    train, test = split_labels(labels)
    assert (len(train), len(test)) == (300, 100)


def test_different_seed_gives_different_split() -> None:
    labels = make_labels([(f"M{i}", "Autre", 1) for i in range(40)])
    _, test_a = split_labels(labels, seed=1)
    _, test_b = split_labels(labels, seed=2)
    assert list(test_a["merchant_key"]) != list(test_b["merchant_key"])


# ---------- running configs ----------


def response_for(batch: list[str], mapping: dict[str, str], confidence: float = 0.9):
    items = [
        {"merchant": key, "category": mapping.get(key, "Autre"), "confidence": confidence}
        for key in batch
    ]
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    block = SimpleNamespace(type="text", text=json.dumps({"items": items}, ensure_ascii=False))
    return SimpleNamespace(content=[block], usage=usage)


class ScriptedClient:
    """Answers every batch from a merchant -> category mapping."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        batch = json.loads(kwargs["messages"][0]["content"])
        return response_for(batch, self.mapping)


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = make_labels([("METRO", "Épicerie", 5), ("MCDONALD'S", "Restaurant", 4)])
    test = make_labels([("NETFLIX.COM", "Abonnements", 2), ("SUPER GYM", "Loisirs", 3)])
    return train, test


def test_rules_only_makes_no_api_call(dataset) -> None:
    train, test = dataset
    client = ScriptedClient({})
    result = run_config("Rules only", train, test, use_llm=False, client=client)

    assert client.calls == []
    assert result.api_calls == 0
    assert result.total == 2
    assert result.accuracy == 0.0  # unknown merchants fall back to "Autre"
    assert result.transactions == 5  # 2 + 3 occurrences


def test_rules_only_scores_train_merchants(dataset) -> None:
    train, _ = dataset
    # A merchant already covered by a training rule needs no API call
    test = make_labels([("METRO", "Épicerie", 2)])
    result = run_config("Rules only", train, test, use_llm=False)
    assert result.accuracy == 1.0
    assert result.api_calls == 0


def test_llm_config_measures_accuracy_cost_and_latency(dataset) -> None:
    train, test = dataset
    client = ScriptedClient({"NETFLIX.COM": "Abonnements", "SUPER GYM": "Autre"})
    result = run_config("+ Haiku", train, test, use_llm=True, client=client)

    assert result.total == 2
    assert result.correct == 1  # SUPER GYM misclassified
    assert result.accuracy == 0.5
    assert result.api_calls == 1  # both merchants fit in one batch
    assert result.cost_usd > 0
    assert result.p50_latency_ms is not None
    assert result.calls_per_1000 == pytest.approx(1 / 5 * 1000)


def test_corrections_config_improves_accuracy(dataset) -> None:
    train, test = dataset
    # The model always answers "Autre": every prediction is wrong…
    client = ScriptedClient({})
    plain = run_config("+ Haiku", train, test, use_llm=True, client=client)
    assert plain.accuracy == 0.0

    # …but after the simulated month the user's corrections are remembered
    corrected = run_config(
        "+ corrections", train, test, use_llm=True, client=ScriptedClient({}),
        simulate_corrections=True,
    )
    assert corrected.accuracy > plain.accuracy
    # Month-1 transactions are counted too, so the cost is honest
    assert corrected.transactions > plain.transactions


def test_predictions_recorded_for_confusion_matrix(dataset) -> None:
    train, test = dataset
    client = ScriptedClient({"NETFLIX.COM": "Abonnements", "SUPER GYM": "Autre"})
    result = run_config("+ Haiku", train, test, use_llm=True, client=client)
    assert sorted(result.predictions) == [
        ("Abonnements", "Abonnements"),
        ("Loisirs", "Autre"),
    ]


# ---------- reporting ----------


def test_confusion_matrix_counts() -> None:
    matrix = confusion_matrix(
        [
            ("Restaurant", "Restaurant"),
            ("Restaurant", "Épicerie"),
            ("Épicerie", "Épicerie"),
        ]
    )
    assert matrix.at["Restaurant", "Restaurant"] == 1
    assert matrix.at["Restaurant", "Épicerie"] == 1
    assert matrix.at["Épicerie", "Épicerie"] == 1
    assert matrix.at["Épicerie", "Restaurant"] == 0


def test_confusion_matrix_only_shows_present_categories() -> None:
    matrix = confusion_matrix([("Restaurant", "Restaurant")])
    assert list(matrix.columns) == ["Restaurant"]


def test_render_markdown_contains_metrics() -> None:
    result = ConfigResult(
        name="+ Haiku",
        correct=90,
        total=100,
        api_calls=4,
        cost_usd=0.02,
        latencies_ms=[500.0, 700.0],
        transactions=1000,
    )
    markdown = render_markdown([result], None, 300, 100)
    assert "| + Haiku | 90.0% | 4.0 | $0.0200 | 600 ms |" in markdown
    assert "seed 42" in markdown


def test_render_markdown_marks_skipped_configs() -> None:
    markdown = render_markdown([ConfigResult(name="+ Sonnet", skipped="no API key")], None, 300, 100)
    assert "| + Sonnet | _no API key_ | — | — | — |" in markdown


def test_update_readme_replaces_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nintro\n\n<!-- EVAL_START -->\nold results\n<!-- EVAL_END -->\n\nfooter\n",
        encoding="utf-8",
    )
    update_readme("new results", readme_path=readme)
    text = readme.read_text(encoding="utf-8")

    assert "new results" in text
    assert "old results" not in text
    assert text.startswith("# Title")
    assert text.rstrip().endswith("footer")


def test_project_readme_has_markers() -> None:
    # Without them the harness would append a duplicate section every run
    from eval.run_eval import END_MARKER, README_FILE, START_MARKER

    text = README_FILE.read_text(encoding="utf-8")
    assert text.count(START_MARKER) == 1
    assert text.count(END_MARKER) == 1


def test_update_readme_is_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n", encoding="utf-8")
    update_readme("results", readme_path=readme)
    once = readme.read_text(encoding="utf-8")
    update_readme("results", readme_path=readme)
    assert readme.read_text(encoding="utf-8") == once
    assert once.count("<!-- EVAL_START -->") == 1
