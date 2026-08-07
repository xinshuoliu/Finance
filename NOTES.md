# Finance App — Project Notes

Personal finance dashboard: upload a bank CSV export, categorize expenses, track budgets and spot recurring payments/subscriptions.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run main.py
```

The app opens in your browser (usually http://localhost:8501). Then upload a CSV export from your bank.

## Tools / stack

| Tool | Role |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework (the whole interface, no HTML needed) |
| [pandas](https://pandas.pydata.org) | Reading CSVs, filtering, grouping, date/amount parsing |
| [Plotly Express](https://plotly.com/python/plotly-express/) | Pie chart of expenses by category |
| JSON files | Simple persistence (no database) |

## Project files

| File | Purpose |
|---|---|
| `main.py` | The whole app (UI + CSV import + logic) |
| `categories.json` | Your categories and the merchant keywords learned for each one |
| `budgets.json` | Budget amount (CAD) per category |
| `recurring.json` | Keywords used to match recurring payments/subscriptions |
| `requirements.txt` | Python dependencies |

The JSON files are created/updated automatically by the app — you normally never edit them by hand (but you can; they are plain JSON).

---

## Features

### 1. CSV import (multi-bank)
- **Auto-detection** of the column layout. It recognizes common column names in English and French: date (`Date`, `Transaction Date`, `Date d'opération`…), description (`Description`, `Details`, `Libellé`…), and amount (`Amount`, `Montant`…) or separate `Debit`/`Credit` columns.
- **Handles messy bank exports**: extra header/preamble lines above the real header (up to 6 skipped automatically), comma / semicolon / tab separators, UTF-8 / Windows-1252 / Latin-1 encodings (so French accents work).
- **Amount formats**: `$1,234.56`, `1 234,56` (French decimal comma), `(123.45)` (parentheses = negative), currency symbols and spaces are cleaned automatically.
- **Date formats**: ISO (`2024-01-15`), `DD/MM/YYYY`, `MM/DD/YYYY`… If the file uses day-first dates, that is detected automatically when possible.
- **Manual mapping fallback**: if auto-detection fails, the app shows dropdowns to pick which column is the date, description and amount — choose them, click **Apply mapping**, and the mapping is remembered for that file for the rest of the session.

**Sign convention (single amount column):** a *positive* amount is treated as a **Debit** (expense/charge) and a *negative* amount as a **Credit** (payment/refund). This matches typical credit-card statements. If your bank uses the opposite convention, that's a known limitation for now (see backlog).

### 2. Filters
- Date range (defaults to the full range in the file)
- Show/hide Debits and Credits
- Free-text search in the transaction details

Filters apply to all tabs (the Recurring tab has a checkbox to opt in/out of them).

### 3. Expenses tab (Debits)
- **Categories with keyword learning**: create a category, then change a transaction's category in the table and click **Apply Changes**. The transaction's description is saved as a keyword for that category, so the same merchant is categorized automatically on every future upload.
- **Expense Summary**: total spent per category.
- **Budgets**: set a budget per category (in CAD) for the filtered period. Budget Status shows a progress bar per category, with over/under warnings.
- **Pie chart** of expenses by category.

### 4. Payments tab (Credits)
- Total of incoming payments and the list of credit transactions.

### 5. Recurring tab
- **Keyword-based tracking**: add keywords (`netflix`, `gym`, `bell`…); any debit whose description contains a keyword is listed, with a per-merchant summary (occurrences, total, average, estimated monthly cost).
- **Auto-detected candidates**: merchants that appear in several distinct months (threshold adjustable with the slider) — copy a name from that table into the keyword box to start tracking it.

---

## How to use (typical workflow)

1. Download a CSV statement from your bank.
2. `streamlit run main.py` and upload the file.
3. If auto-detection fails, map the columns manually and click **Apply mapping**.
4. In **Expenses**, create your categories, then assign categories to transactions and click **Apply Changes**. The app learns the merchants — next month's upload will mostly categorize itself.
5. Set budgets per category and watch the Budget Status bars.
6. In **Recurring**, check the auto-detected candidates and add keywords for your subscriptions.

## How the data works

- Nothing is uploaded anywhere: everything runs locally, the CSV stays in memory for the session.
- Only your *rules* are persisted (categories/keywords, budgets, recurring keywords) in the JSON files — the transactions themselves are re-read from the CSV each time.
- Keyword matching is case-insensitive substring matching on the transaction description.
- If two categories share a matching keyword, the last category (in `categories.json` order) wins.

---

## Known limitations

- Transactions are not stored — you re-upload the CSV each session.
- One file at a time (no merging of multiple accounts/statements).
- Sign convention for single-amount CSVs is fixed (positive = expense).
- Category keyword = the full transaction description (exact merchant string), so a merchant with a slightly different description string is not matched.
- Budgets apply to "the current filtered period", not per-month.

## Feature ideas / backlog

- [ ] Toggle for the amount sign convention (bank account vs credit card exports)
- [ ] Persist transactions (SQLite or parquet) and merge multiple uploads
- [ ] Monthly budgets with month-by-month history
- [ ] Edit/delete categories and their keywords from the UI
- [ ] Monthly spending trend chart (bar/line over time)
- [ ] Export categorized data to CSV/Excel
- [ ] Multi-currency support

---

## AI layer (in progress)

An AI layer is being added on top of the app: merchant normalization + cache,
cascading categorizer (cache → rules → Claude Haiku), review queue with rule
learning, natural-language queries via validated filter specs, French monthly
narrative (Claude Sonnet), statistical recurring detection, and an evaluation
harness (`eval/`). Key design decisions:

- **Closed category set (French)**: Épicerie, Restaurant, Achats, Transport,
  Logement, Services publics, Santé, Loisirs, Abonnements, Revenu, Transfert,
  Autre. ("Achats" was added to the original 11 — the data needs a general
  shopping bucket.) Existing user categories and budgets get migrated in Phase 2.
- **Privacy**: only *normalized merchant strings* are ever sent to the API for
  categorization — never amounts, balances, dates tied to amounts, or account
  numbers. The monthly narrative receives pre-computed aggregates only. The
  model never produces numbers; all math happens in pandas.
- **Works without a key**: every AI feature degrades gracefully when
  `ANTHROPIC_API_KEY` is missing (e.g. on the deployed Streamlit Cloud app).
  Local AI state lives in `data/` (gitignored). API key goes in `.env`
  (gitignored) as `ANTHROPIC_API_KEY=...`.
- **Models**: `claude-haiku-4-5` for merchant classification (cheap, batched
  ~30/request), `claude-sonnet-5` for the monthly narrative. Pricing constants
  live in one marked block in `ai/config.py`.
- **Language**: the UI, docstrings and narrative are in **English** (user
  decision, 2026-08-07). Only the category *labels* stay French — they are
  the spec's closed set and all stored data (rules, cache, budgets) is keyed
  on them. The categorizer prompt is French on purpose (Quebec merchants).

**Phase 1 (done)** — `ai/normalize.py` collapses bank description variants to
stable merchant keys (strips ref numbers, `#`/`*` suffixes, POS/ACHAT/INTERAC
tokens, trailing city+province); `ai/cache.py` persists merchant → category in
`data/merchant_cache.json` (atomic writes, corruption-tolerant);
`ai/config.py` holds models/pricing/thresholds. Tests: `python -m pytest`
(offline, no API calls).

**Phase 2 (done)** — Cascading categorizer wired into the dashboard:

- `ai/categorize.py` — cascade **cache → rules → Claude Haiku** (batches of
  ~30 merchants, structured outputs lock the category to the closed set,
  one retry on invalid output, `Autre`+`à revoir` fallback). API transport
  failures are *not* cached so they retry later; model verdicts are.
- `ai/privacy.py` — the only builder of outbound payloads; accepts merchant
  strings only, anything else raises `TypeError` (unit-tested: no amount can
  reach the API by construction).
- `ai/llm_log.py` — every API call goes through `logged_call()`: model,
  tokens, latency, prompt-cache stats and cost (from the pricing block) are
  appended to `data/api_log.jsonl`.
- `ai/rules.py` — deterministic rules in `data/category_rules.json`
  (substring or regex, first match wins, user rules take precedence).
- `ai/migrate.py` — one-time migration: legacy learned keywords →
  normalized rules with French categories; budgets re-keyed
  (restaurant→Restaurant, gym+Activities→Loisirs…). Runs automatically at
  first app start; the existence of the rules file marks it done.
- `main.py` — the old keyword system is gone. Transactions get
  `Category`/`Confidence`/`Source`/`NeedsReview` columns; a caption shows how
  many came from cache/rules/AI; correcting a category in the table creates
  a rule + cache entry so the merchant stays fixed forever. Category list
  is now the closed French set (also for budgets).

**Phase 3 (done)** — Review queue and rule learning:

- New **Review** tab: merchants the AI wasn't sure about (`needs_review`),
  grouped per merchant with transaction count, total and confidence. Pick
  the right category and *Apply corrections* (changed rows only) or
  *Confirm all shown suggestions* (records every shown row as user-confirmed).
- Every correction does three things: updates the transactions, writes a
  `source: "user"` rule to `data/category_rules.json`, and re-applies it —
  the success message shows "N similar transactions updated (M new rules)".
- User rules are inserted *before* legacy rules and override a stale cache
  entry, so a correction always wins.
- Corrections survive re-importing the same statement (and API downtime) —
  proven by `tests/test_review_flow.py`.

**Phase 4 (done)** — Natural-language queries (new **Ask** tab):

- Type a question ("How much did I spend on restaurants since January?" —
  English or French). Claude Haiku translates it into a **filter spec, never
  code**: `{categories, date_from, date_to, aggregate, group_by,
  merchant_contains, transaction_type}`.
- Double validation: structured outputs lock the JSON schema (categories and
  enums), then a Pydantic model (`ai/query.py::FilterSpec`) re-validates —
  closed-set categories, ISO dates in order, `aggregate` ∈ {sum, mean, count,
  max}, `group_by` ∈ {month, week, category, merchant, null}. Anything else
  is rejected with a clear user-facing message.
- The spec runs locally in pandas (`execute_spec`); grouped results render
  as a Plotly bar chart + table, scalar results as a metric with the matching
  transactions expandable. The interpreted filter is always shown above the
  result so you can see what was understood.
- Relative dates ("since January", "last month") resolve against today's
  date, which is included in the prompt.

---

## Changelog

- **2026-08-07** — AI layer Phase 4: Ask tab — natural-language questions
  translated to Pydantic-validated filter specs (never code), executed in
  pandas, rendered with Plotly.
- **2026-08-07** — AI layer Phase 3: Review tab (review queue with per-merchant
  corrections and "N similar transactions updated" feedback); all UI strings
  and docstrings switched to English; re-import survival tests.
- **2026-08-07** — AI layer Phase 2: cascading categorizer (cache → rules →
  Claude Haiku) wired into the dashboard; privacy guard; API call logging;
  legacy keywords/budgets migrated to the closed French category set.
- **2026-08-07** — AI layer Phase 1: merchant normalization (`ai/normalize.py`),
  persistent categorization cache (`ai/cache.py`), AI config with pricing
  constants (`ai/config.py`), offline pytest suite (`tests/`).
- **2026-08-07** — Functionality fix pass:
  - Budgets, Budget Status and the pie chart were rendering *below* the tabs instead of inside the Expenses tab (indentation bug) — moved into the tab.
  - Crash fixed: selecting a single date in the range picker (mid-selection) crashed the app.
  - Crash fixed: a CSV that parsed to zero valid rows crashed on the date filter; now shows an error message.
  - Manual column mapping now has an **Apply mapping** button and is remembered per file (before, it applied default columns immediately and produced wrong/empty data).
  - CSV reading now tries multiple encodings (UTF-8, Windows-1252, Latin-1) and separators (`,` `;` tab) — French exports with accents no longer fail.
  - French amount format `1 234,56` was parsed as `123456` — fixed proper decimal-comma handling, plus parentheses negatives.
  - Day-first date detection (`25/03/2024` no longer risks being read as month 25 → dropped or swapped).
  - Empty rows (no amount) in debit/credit-split CSVs are dropped instead of appearing as 0.00 debits.
  - Applying category changes / adding-removing recurring keywords now refreshes the page immediately (tables and summaries update at once).
  - Removed deprecated pandas/Streamlit API usage (`infer_datetime_format`, `use_container_width`) that spammed warnings.
  - Corrupt `categories.json` no longer crashes the app on startup.
  - `README.md` had broken text encoding — rewritten.
