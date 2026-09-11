"""Evaluation harness: accuracy, cost and latency of the categorization cascade.

Two modes:

    python -m eval.run_eval --make-template statement1.csv statement2.csv
        Build eval/labels.csv with one row per distinct merchant (normalized
        and deduplicated), ready to hand-label. Existing labels are kept.

    python -m eval.run_eval
        Run the labelled set through every configuration on a fixed, seeded
        train/test split and rewrite the results table in README.md.

Every configuration starts from an empty cache and its own rule store, so the
numbers are not contaminated by earlier runs or by the app's own state.
"""

import argparse
import json
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import ai.llm_log as llm_log
from ai.categorize import categorize_keys, record_user_correction
from ai.cache import MerchantCache
from ai.config import CATEGORIES, MODEL_CATEGORIZER, MODEL_NARRATIVE, ai_available
from ai.normalize import normalize
from ai.rules import RuleStore
from importer import load_bank_csv

EVAL_DIR = Path(__file__).resolve().parent
LABELS_FILE = EVAL_DIR / "labels.csv"
README_FILE = EVAL_DIR.parent / "README.md"

SEED = 42
TRAIN_SIZE = 300
TEST_SIZE = 100

START_MARKER = "<!-- EVAL_START -->"
END_MARKER = "<!-- EVAL_END -->"

LABEL_COLUMNS = ["merchant_key", "category", "occurrences", "example_description"]


# --------------------------------------------------------------------------
# Labels template
# --------------------------------------------------------------------------


def _read_labels_csv(path: Path) -> pd.DataFrame:
    """Read a labels file, tolerating what Excel writes on Windows."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path.name}: could not decode the file")


def build_template(csv_paths: list[str], labels_path: Path = LABELS_FILE) -> pd.DataFrame:
    """Build (or refresh) the labelling template from bank CSV exports.

    One row per distinct normalized merchant key, with its transaction count
    and an example description. Categories already filled in are preserved.
    """
    frames = [load_bank_csv(path) for path in csv_paths]
    if not frames:
        raise ValueError("no CSV file provided")

    df = pd.concat(frames, ignore_index=True)
    df["merchant_key"] = df["Details"].map(normalize)
    df = df[df["merchant_key"] != ""]

    template = (
        df.groupby("merchant_key")
        .agg(occurrences=("Details", "size"), example_description=("Details", "first"))
        .reset_index()
        .sort_values(["occurrences", "merchant_key"], ascending=[False, True])
    )
    template["category"] = ""

    if labels_path.exists():
        existing = _read_labels_csv(labels_path).fillna({"category": ""})
        known = dict(zip(existing["merchant_key"], existing["category"]))
        template["category"] = template["merchant_key"].map(lambda k: known.get(k, ""))

    template = template[LABEL_COLUMNS]
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(labels_path, index=False, encoding="utf-8-sig")
    return template


def load_labels(labels_path: Path = LABELS_FILE) -> pd.DataFrame:
    """Load the labelled rows only; raise ValueError on unknown categories."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found — run: python -m eval.run_eval --make-template <csv>…"
        )
    df = _read_labels_csv(labels_path).fillna({"category": ""})
    df["category"] = df["category"].astype(str).str.strip()
    labelled = df[df["category"] != ""].copy()

    unknown = sorted(set(labelled["category"]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories in {labels_path.name}: {', '.join(unknown)}")

    if "occurrences" not in labelled.columns:
        labelled["occurrences"] = 1
    labelled["occurrences"] = labelled["occurrences"].fillna(1).astype(int)
    return labelled.reset_index(drop=True)


def split_labels(
    labels: pd.DataFrame, seed: int = SEED, train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle deterministically and split into train/test.

    With fewer labels than train_size + test_size, the same 75/25 proportion
    is kept so a partially-labelled file still produces a usable report.
    """
    order = list(range(len(labels)))
    random.Random(seed).shuffle(order)
    shuffled = labels.iloc[order].reset_index(drop=True)

    total = len(shuffled)
    if total >= train_size + test_size:
        n_train, n_test = train_size, test_size
    else:
        n_test = max(1, round(total * test_size / (train_size + test_size)))
        n_train = total - n_test

    train = shuffled.iloc[:n_train].reset_index(drop=True)
    test = shuffled.iloc[n_train : n_train + n_test].reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class ConfigResult:
    """Measured outcome of one configuration on the held-out set."""

    name: str
    correct: int = 0
    total: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    transactions: int = 0
    predictions: list[tuple[str, str]] = field(default_factory=list)  # (true, predicted)
    skipped: str | None = None  # reason, when the config could not run

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def calls_per_1000(self) -> float:
        return self.api_calls / self.transactions * 1000 if self.transactions else 0.0

    @property
    def cost_per_1000(self) -> float:
        return self.cost_usd / self.transactions * 1000 if self.transactions else 0.0

    @property
    def p50_latency_ms(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None


def _log_offset() -> int:
    path = llm_log.API_LOG_FILE
    return path.stat().st_size if path.exists() else 0


def _read_log_since(offset: int) -> tuple[list[dict], int]:
    """Read the API log records appended since a byte offset."""
    path = llm_log.API_LOG_FILE
    if not path.exists():
        return [], offset
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offset)
        records = [json.loads(line) for line in f if line.strip()]
        return records, f.tell()


def _fresh_stores(train: pd.DataFrame) -> tuple[MerchantCache, RuleStore]:
    """Empty cache plus a rule store holding the training merchants."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_"))
    cache = MerchantCache(path=tmp / "cache.json")
    rules = RuleStore(path=tmp / "rules.json")
    for row in train.itertuples():
        rules.add(row.merchant_key, row.category, source="train")
    return cache, rules


def _score(
    result: ConfigResult, keys: list[str], truth: dict[str, str], predictions: dict
) -> None:
    for key in keys:
        predicted = predictions[key].category
        result.predictions.append((truth[key], predicted))
        result.total += 1
        if predicted == truth[key]:
            result.correct += 1


def _account(result: ConfigResult, offset: int) -> int:
    """Fold the API records logged since `offset` into the result."""
    records, new_offset = _read_log_since(offset)
    for record in records:
        if record.get("purpose") != "categorize":
            continue
        result.api_calls += 1
        result.cost_usd += float(record.get("cost_usd") or 0.0)
        if record.get("latency_ms") is not None:
            result.latencies_ms.append(float(record["latency_ms"]))
    return new_offset


def run_config(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_llm: bool,
    model: str = MODEL_CATEGORIZER,
    client=None,
    simulate_corrections: bool = False,
    seed: int = SEED,
) -> ConfigResult:
    """Run one configuration over the held-out set and measure it."""
    result = ConfigResult(name=name)
    cache, rules = _fresh_stores(train)
    truth = dict(zip(test["merchant_key"], test["category"]))
    occurrences = dict(zip(test["merchant_key"], test["occurrences"]))
    test_keys = list(test["merchant_key"])
    offset = _log_offset()

    if simulate_corrections:
        # Month 1: a random half of the merchants appear; the user fixes every
        # mistake in the review queue, which writes rules and cache entries.
        month1 = test_keys[:]
        random.Random(seed + 1).shuffle(month1)
        month1 = month1[: max(1, len(month1) // 2)]

        first_pass = categorize_keys(month1, cache, rules, client=client, use_llm=True, model=model)
        for key in month1:
            if first_pass[key].category != truth[key]:
                record_user_correction(key, truth[key], cache, rules, save=False)
        result.transactions += sum(occurrences[key] for key in month1)
        offset = _account(result, offset)

    predictions = categorize_keys(
        test_keys, cache, rules, client=client, use_llm=use_llm, model=model
    )
    result.transactions += sum(occurrences.values())
    _account(result, offset)
    _score(result, test_keys, truth, predictions)
    return result


def confusion_matrix(predictions: list[tuple[str, str]]) -> pd.DataFrame:
    """Rows = true category, columns = predicted category."""
    labels = [c for c in CATEGORIES if any(c in pair for pair in predictions)]
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for true, predicted in predictions:
        if true in matrix.index and predicted in matrix.columns:
            matrix.at[true, predicted] += 1
    return matrix


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_markdown(
    results: list[ConfigResult], matrix: pd.DataFrame | None, train_n: int, test_n: int
) -> str:
    """Render the results table (and confusion matrix) as markdown."""
    lines = [
        f"_Seeded {train_n}/{test_n} train/test split (seed {SEED}), measured on the "
        "held-out set. Regenerate with `python -m eval.run_eval`._",
        "",
        "| Config | Accuracy | API calls / 1000 txns | Cost / 1000 txns | p50 latency |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| {r.name} | _{r.skipped}_ | — | — | — |")
            continue
        latency = f"{r.p50_latency_ms:.0f} ms" if r.p50_latency_ms is not None else "—"
        lines.append(
            f"| {r.name} | {r.accuracy:.1%} | {r.calls_per_1000:.1f} | "
            f"${r.cost_per_1000:.4f} | {latency} |"
        )

    if matrix is not None and not matrix.empty:
        lines += ["", f"**Confusion matrix — {matrix.attrs.get('config', 'model')}** "
                  "(rows = true, columns = predicted)", ""]
        header = "| true \\ predicted | " + " | ".join(matrix.columns) + " |"
        lines.append(header)
        lines.append("|---" * (len(matrix.columns) + 1) + "|")
        for label, row in matrix.iterrows():
            cells = " | ".join(str(v) if v else "·" for v in row)
            lines.append(f"| **{label}** | {cells} |")

    return "\n".join(lines)


def update_readme(markdown: str, readme_path: Path = README_FILE) -> None:
    """Write the report between the EVAL markers, adding the section if absent."""
    block = f"{START_MARKER}\n{markdown}\n{END_MARKER}"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Finance App\n"

    if START_MARKER in text and END_MARKER in text:
        head = text.split(START_MARKER)[0]
        tail = text.split(END_MARKER, 1)[1]
        text = f"{head}{block}{tail}"
    else:
        text = f"{text.rstrip()}\n\n## Categorization evaluation\n\n{block}\n"

    readme_path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Category names carry accents; a cp1252 console would garble or crash
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--make-template",
        nargs="+",
        metavar="CSV",
        help="build eval/labels.csv from bank CSV exports, then exit",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-readme", action="store_true", help="print the report without writing README.md")
    args = parser.parse_args(argv)

    if args.make_template:
        template = build_template(args.make_template)
        todo = int((template["category"] == "").sum())
        print(f"Wrote {LABELS_FILE} — {len(template)} merchants, {todo} still to label.")
        print("Fill in the 'category' column with one of:")
        print("  " + ", ".join(CATEGORIES))
        return 0

    try:
        labels = load_labels()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(labels) < 8:
        print(
            f"Only {len(labels)} labelled merchants in {LABELS_FILE.name} — "
            "label more rows before running the evaluation.",
            file=sys.stderr,
        )
        return 1

    train, test = split_labels(labels, seed=args.seed)
    print(f"{len(labels)} labelled merchants — {len(train)} train / {len(test)} test\n")

    results = [run_config("Rules only", train, test, use_llm=False, seed=args.seed)]

    if not ai_available():
        for name in (f"+ Haiku ({MODEL_CATEGORIZER})", f"+ Sonnet ({MODEL_NARRATIVE})",
                     "+ corrections after simulated month"):
            results.append(ConfigResult(name=name, skipped="no API key"))
    else:
        results.append(
            run_config(f"+ Haiku ({MODEL_CATEGORIZER})", train, test, use_llm=True,
                       model=MODEL_CATEGORIZER, seed=args.seed)
        )
        results.append(
            run_config(f"+ Sonnet ({MODEL_NARRATIVE})", train, test, use_llm=True,
                       model=MODEL_NARRATIVE, seed=args.seed)
        )
        results.append(
            run_config("+ corrections after simulated month", train, test, use_llm=True,
                       model=MODEL_CATEGORIZER, simulate_corrections=True, seed=args.seed)
        )

    matrix = None
    for result in results:
        if result.skipped is None and result.name.startswith("+ Haiku"):
            matrix = confusion_matrix(result.predictions)
            matrix.attrs["config"] = result.name
    if matrix is None:
        matrix = confusion_matrix(results[0].predictions)
        matrix.attrs["config"] = results[0].name

    markdown = render_markdown(results, matrix, len(train), len(test))
    print(markdown)

    if not args.no_readme:
        update_readme(markdown)
        print(f"\nResults written to {README_FILE.name} between the EVAL markers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
