# Finance App

Personal finance dashboard built with Streamlit: upload a bank CSV, categorize expenses with a cache → rules → Claude cascade, set budgets, ask questions in plain English, and track recurring payments.

**Live app:** https://urfinance.streamlit.app/

```bash
pip install -r requirements.txt
streamlit run main.py
```

AI features need an Anthropic API key in a `.env` file (`ANTHROPIC_API_KEY=sk-ant-...`); without one the app runs with AI disabled.

Full documentation (features, tools, how to use, backlog): see [NOTES.md](NOTES.md).

## Categorization evaluation

```bash
python -m eval.run_eval --make-template statement1.csv statement2.csv  # build eval/labels.csv
python -m eval.run_eval                                                # measure, rewrite the table below
```

<!-- EVAL_START -->
_No evaluation has been run yet. Label `eval/labels.csv` and run `python -m eval.run_eval` to fill in this table._
<!-- EVAL_END -->
