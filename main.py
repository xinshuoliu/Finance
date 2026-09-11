import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from ai.cache import MerchantCache
from ai.categorize import Categorization, categorize_keys, record_user_correction
from ai.config import CATEGORIES, CATEGORY_RULES_FILE, ai_available
from ai.migrate import migrate_legacy
from ai.narrative import NarrativeError, build_month_summary, generate_narrative
from ai.normalize import normalize
from ai.query import FilterSpec, QueryTranslationError, execute_spec, translate_question
from ai.rules import RuleStore
from analysis import detect_recurring
from importer import auto_detect_mapping, mapping_matches, normalize_transactions, smart_read_csv

st.set_page_config(page_title="Simple Finance App", page_icon="💰", layout="wide")

category_file = "categories.json"
budget_file = "budgets.json"
recurring_file = "recurring.json"


@st.cache_resource
def get_ai_stores() -> tuple[RuleStore, MerchantCache]:
    """Load rules and merchant cache; migrate the legacy system on first launch."""
    run_migration = not CATEGORY_RULES_FILE.exists() and os.path.exists(category_file)
    rules = RuleStore()
    cache = MerchantCache()
    if run_migration:
        migrate_legacy(category_file, budget_file, rules)
    return rules, cache


rules_store, merchant_cache = get_ai_stores()

if "budgets" not in st.session_state:
    st.session_state.budgets = {}
    if os.path.exists(budget_file):
        try:
            with open(budget_file, "r", encoding="utf-8") as f:
                st.session_state.budgets = json.load(f)
        except Exception:
            st.session_state.budgets = {}

if "recurring" not in st.session_state:
    st.session_state.recurring = []
    if os.path.exists(recurring_file):
        try:
            with open(recurring_file, "r", encoding="utf-8") as f:
                st.session_state.recurring = json.load(f)
        except Exception:
            st.session_state.recurring = []


def save_budgets():
    with open(budget_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.budgets, f, ensure_ascii=False)


def save_recurring():
    with open(recurring_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.recurring, f, ensure_ascii=False)


def apply_ai_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize the DataFrame through the cache -> rules -> AI cascade."""
    df = df.copy()
    df["Merchant"] = df["Details"].map(normalize)
    use_llm = ai_available() and not st.session_state.get("ai_llm_down", False)
    results = categorize_keys(df["Merchant"].tolist(), merchant_cache, rules_store, use_llm=use_llm)

    default = Categorization("Autre", 0.0, "fallback", True)
    df["Category"] = [results.get(k, default).category for k in df["Merchant"]]
    df["Confidence"] = [results.get(k, default).confidence for k in df["Merchant"]]
    df["Source"] = [results.get(k, default).source for k in df["Merchant"]]
    df["NeedsReview"] = [results.get(k, default).needs_review for k in df["Merchant"]]

    if use_llm and any(r.source == "fallback" for r in results.values()):
        # The API failed this run: stop retrying on every rerun of this session
        st.session_state.ai_llm_down = True
    return df


def _mapping_ui(df: pd.DataFrame) -> dict | None:
    st.warning("Couldn't auto-detect your bank's CSV format. Map the columns manually, then click Apply.")
    cols = list(df.columns)

    date_col = st.selectbox("Date column", options=cols, key="map_date")
    details_col = st.selectbox("Details / Description column", options=cols, key="map_details")

    mode = st.radio(
        "Amount format",
        ["Single amount column (sign indicates debit/credit)", "Separate Debit and Credit columns"],
        key="map_mode",
    )

    if mode.startswith("Single"):
        amount_col = st.selectbox("Amount column", options=cols, key="map_amount")
        mapping = {
            "mode": "single",
            "date_col": date_col,
            "details_col": details_col,
            "amount_col": amount_col,
        }
    else:
        debit_col = st.selectbox("Debit column (money out)", options=cols, key="map_debit")
        credit_col = st.selectbox("Credit column (money in)", options=cols, key="map_credit")
        mapping = {
            "mode": "split",
            "date_col": date_col,
            "details_col": details_col,
            "debit_col": debit_col,
            "credit_col": credit_col,
        }

    if st.button("Apply mapping", type="primary"):
        return mapping
    return None


def load_transactions(file):
    try:
        df_raw = smart_read_csv(file)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    mapping = auto_detect_mapping(df_raw)

    # ---- Manual mapping fallback (persisted per file) ----
    if mapping is None:
        stored = st.session_state.get("manual_mapping")
        if stored and stored.get("file") == file.name and mapping_matches(stored["mapping"], df_raw):
            mapping = stored["mapping"]
        else:
            mapping = _mapping_ui(df_raw)
            if mapping is None:
                return None
            st.session_state.manual_mapping = {"file": file.name, "mapping": mapping}

    try:
        df = normalize_transactions(df_raw, mapping)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    if df.empty:
        st.error("No valid transactions found in this file. Check the column mapping and the file contents.")
        st.session_state.pop("manual_mapping", None)
        return None

    return df


def main():
    st.title("Finance Dashboard")

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a bank CSV export to get started.")
        return

    df = load_transactions(uploaded_file)
    if df is None:
        return

    with st.spinner("Categorizing transactions…"):
        df = apply_ai_categories(df)

    if ai_available():
        source_counts = df["Source"].value_counts()
        st.caption(
            "🤖 Categorization — "
            f"cache: {int(source_counts.get('cache', 0))}, "
            f"rules: {int(source_counts.get('rule', 0))}, "
            f"AI: {int(source_counts.get('llm', 0))}, "
            f"needs review: {int(df['NeedsReview'].sum())}"
        )
        if st.session_state.get("ai_llm_down"):
            st.warning(
                "The Anthropic API call failed — unknown merchants are filed "
                "under \"Autre\" for now."
            )
            if st.button("Retry AI"):
                st.session_state.ai_llm_down = False
                st.rerun()
    else:
        st.caption(
            "🤖 AI disabled — add ANTHROPIC_API_KEY to a .env file to enable "
            "automatic categorization."
        )

    st.subheader("Filters")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        picked = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        # date_input returns a 1-tuple while the user is mid-selection
        if isinstance(picked, (tuple, list)):
            start_date = picked[0]
            end_date = picked[1] if len(picked) > 1 else picked[0]
        else:
            start_date = end_date = picked

    with col2:
        show_debits = st.checkbox("Show Debits", value=True)
        show_credits = st.checkbox("Show Credits", value=True)

    with col3:
        search_text = st.text_input("Search in Details").lower().strip()

    filtered_df = df.copy()

    # date filter
    filtered_df = filtered_df[
        (filtered_df["Date"].dt.date >= start_date) &
        (filtered_df["Date"].dt.date <= end_date)
    ]

    # debit / credit filter
    allowed_types = []
    if show_debits:
        allowed_types.append("Debit")
    if show_credits:
        allowed_types.append("Credit")

    filtered_df = filtered_df[filtered_df["Debit/Credit"].isin(allowed_types)]

    # search filter
    if search_text:
        filtered_df = filtered_df[
            filtered_df["Details"].astype(str).str.lower().str.contains(search_text, na=False)
        ]

    debits_df = filtered_df[filtered_df["Debit/Credit"] == "Debit"].copy()
    credits_df = filtered_df[filtered_df["Debit/Credit"] == "Credit"].copy()

    st.session_state.debits_df = debits_df.copy()

    detected_recurring = detect_recurring(df)

    tab1, tab2, tab_review, tab3, tab_report, tab_ask = st.tabs(
        ["Expenses (Debits)", "Payments (Credits)", "Review", "Recurring", "Report", "Ask"]
    )

    with tab1:
        st.subheader("Your Expenses")
        if message := st.session_state.pop("correction_message", None):
            st.success(message)
        st.caption(
            "Fix a category and click Apply Changes: a rule is created and will "
            "apply to this merchant in every future statement."
        )
        edited_df = st.data_editor(
            st.session_state.debits_df[["Date", "Details", "Amount", "Category", "Confidence", "NeedsReview"]],
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD"),
                "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0
                ),
                "NeedsReview": st.column_config.CheckboxColumn("Review"),
            },
            disabled=["Date", "Details", "Amount", "Confidence", "NeedsReview"],
            hide_index=True,
            width="stretch",
            key="category_editor",
        )
        save_button = st.button("Apply Changes", type="primary")
        if save_button:
            corrected_merchants = 0
            matching_transactions = 0
            for idx, row in edited_df.iterrows():
                chosen_category = row["Category"]
                if chosen_category == st.session_state.debits_df.at[idx, "Category"]:
                    continue
                merchant_key = st.session_state.debits_df.at[idx, "Merchant"]
                if not record_user_correction(
                    merchant_key, chosen_category, merchant_cache, rules_store, save=False
                ):
                    continue
                corrected_merchants += 1
                matching_transactions += int((df["Merchant"] == merchant_key).sum())
            if corrected_merchants:
                rules_store.save()
                merchant_cache.save()
                st.session_state.correction_message = (
                    f"{matching_transactions} similar transactions updated "
                    f"({corrected_merchants} new rules)."
                )
                st.rerun()

        st.subheader("Expense Summary")
        category_totals = st.session_state.debits_df.groupby("Category")["Amount"].sum().reset_index()
        category_totals = category_totals.sort_values("Amount", ascending=False)

        st.dataframe(
            category_totals,
            column_config={
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD")
            },
            width="stretch",
            hide_index=True,
        )

        st.subheader("Budgets (per category)")

        selected_cat = st.selectbox("Choose a category", options=CATEGORIES)

        current_budget = float(st.session_state.budgets.get(selected_cat, 0) or 0.0)
        new_budget = st.number_input(
            "Budget (CAD) for the current filtered period",
            min_value=0.0,
            value=current_budget,
            step=10.0,
        )

        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("Save Budget"):
                st.session_state.budgets[selected_cat] = float(new_budget)
                save_budgets()
                st.success(f"Budget saved for {selected_cat}.")
        with colB:
            if st.button("Clear Budget"):
                if selected_cat in st.session_state.budgets:
                    del st.session_state.budgets[selected_cat]
                    save_budgets()
                st.info(f"Budget cleared for {selected_cat}.")

        st.subheader("Budget Status")

        for _, row in category_totals.iterrows():
            cat = row["Category"]
            spent = float(row["Amount"])
            budget = float(st.session_state.budgets.get(cat, 0) or 0.0)

            if budget > 0:
                ratio = spent / budget

                if spent > budget:
                    st.warning(f"⚠️ {cat}: {spent:.2f} CAD spent (budget {budget:.2f} CAD)")
                    st.write(f"Over by: {(spent - budget):.2f} CAD")
                else:
                    remaining = budget - spent
                    st.success(f"✅ {cat}: {spent:.2f} CAD spent — {remaining:.2f} CAD remaining")
                    st.write(f"Used: {ratio*100:.0f}%")

                st.progress(min(ratio, 1.0))

        if not category_totals.empty:
            fig = px.pie(
                category_totals,
                values="Amount",
                names="Category",
                title="Expenses by Category",
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader("Payment Summary")
        total_payments = credits_df["Amount"].sum()
        st.metric("Total Payments", f"{total_payments:,.2f} CAD")
        st.dataframe(
            credits_df[["Date", "Details", "Amount"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD"),
            },
        )

    with tab_review:
        st.subheader("Review queue")

        if message := st.session_state.pop("review_message", None):
            st.success(message)

        review_df = df[df["NeedsReview"]].copy()
        if review_df.empty:
            st.success("Nothing to review — every merchant is categorized with good confidence.")
        else:
            review_summary = (
                review_df.groupby("Merchant")
                .agg(
                    Transactions=("Merchant", "size"),
                    Total=("Amount", "sum"),
                    Confidence=("Confidence", "max"),
                    Category=("Category", "first"),
                )
                .reset_index()
                .sort_values("Total", ascending=False)
                .reset_index(drop=True)
            )
            st.caption(
                f"{len(review_summary)} merchants ({len(review_df)} transactions) the AI "
                "wasn't sure about. Pick the right category, then apply — each decision "
                "becomes a permanent rule."
            )
            edited_review = st.data_editor(
                review_summary,
                column_config={
                    "Merchant": st.column_config.TextColumn("Merchant"),
                    "Transactions": st.column_config.NumberColumn("Transactions"),
                    "Total": st.column_config.NumberColumn("Total", format="%.2f CAD"),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0
                    ),
                    "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
                },
                disabled=["Merchant", "Transactions", "Total", "Confidence"],
                hide_index=True,
                width="stretch",
                key="review_editor",
            )

            def _apply_review(rows: pd.DataFrame, only_changed: bool) -> None:
                corrected = 0
                matching = 0
                for i, row in rows.iterrows():
                    if only_changed and row["Category"] == review_summary.at[i, "Category"]:
                        continue
                    if not record_user_correction(
                        row["Merchant"], row["Category"], merchant_cache, rules_store, save=False
                    ):
                        continue
                    corrected += 1
                    matching += int((df["Merchant"] == row["Merchant"]).sum())
                if corrected:
                    rules_store.save()
                    merchant_cache.save()
                    st.session_state.review_message = (
                        f"{matching} similar transactions updated ({corrected} new rules)."
                    )
                    st.rerun()
                else:
                    st.info("No category was changed.")

            colR1, colR2 = st.columns([1, 2])
            with colR1:
                if st.button("Apply corrections", type="primary"):
                    _apply_review(edited_review, only_changed=True)
            with colR2:
                if st.button("Confirm all shown suggestions"):
                    _apply_review(edited_review, only_changed=False)

    with tab3:
        st.subheader("Recurring payments / subscriptions")

        st.markdown("**Detected automatically**")
        st.caption(
            "Merchants charging at a steady rhythm (weekly, biweekly, monthly or yearly) "
            "with stable amounts — computed locally with statistics, no AI involved. "
            "Always uses the whole file, ignoring the filters above."
        )
        if detected_recurring.empty:
            st.info(
                "No recurring pattern detected yet — this needs at least 3 charges "
                "of the same merchant at a steady interval."
            )
        else:
            st.dataframe(
                detected_recurring,
                width="stretch",
                hide_index=True,
                column_config={
                    "AverageAmount": st.column_config.NumberColumn("Average", format="%.2f CAD"),
                    "LastDate": st.column_config.DateColumn("Last charge", format="DD/MM/YYYY"),
                    "NextExpected": st.column_config.DateColumn("Next expected", format="DD/MM/YYYY"),
                    "MonthlyEstimate": st.column_config.NumberColumn(
                        "Monthly estimate", format="%.2f CAD"
                    ),
                },
            )

        st.divider()

        use_filtered = st.checkbox("Use current filters (date/search) for recurring view", value=False)
        base_df = filtered_df.copy() if use_filtered else df.copy()

        base_df = base_df[base_df["Debit/Credit"] == "Debit"].copy()

        st.caption("Add keywords like: bell, videotron, gym, spotify, netflix, amazon, internet, etc.")

        colA, colB = st.columns([2, 1])

        with colA:
            new_kw = st.text_input("Add recurring keyword (matches if it appears in Details)")
        with colB:
            if st.button("Add keyword"):
                k = new_kw.strip()
                if k and k.lower() not in [x.lower() for x in st.session_state.recurring]:
                    st.session_state.recurring.append(k)
                    save_recurring()
                    st.rerun()

        if st.session_state.recurring:
            colC, colD = st.columns([2, 1])
            with colC:
                to_remove = st.selectbox("Remove keyword", options=st.session_state.recurring)
            with colD:
                if st.button("Remove selected"):
                    st.session_state.recurring = [x for x in st.session_state.recurring if x != to_remove]
                    save_recurring()
                    st.rerun()

        st.divider()

        st.subheader("Matched recurring transactions")

        if not st.session_state.recurring:
            st.info("Add at least one keyword to see matches.")
        else:
            keywords = [k.lower().strip() for k in st.session_state.recurring if k.strip()]
            matched = base_df[
                base_df["Details"].astype(str).str.lower().apply(lambda s: any(k in s for k in keywords))
            ].copy()

            matched = matched.sort_values("Date", ascending=False)

            st.dataframe(
                matched[["Date", "Details", "Amount"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD"),
                },
            )

            st.subheader("Recurring summary (by merchant)")
            summary = (
                matched.groupby("Details")["Amount"]
                .agg(Occurrences="count", Total="sum", Average="mean")
                .reset_index()
                .sort_values("Total", ascending=False)
            )

            matched["Month"] = matched["Date"].dt.to_period("M").astype(str)
            monthly_totals = (
                matched.groupby(["Details", "Month"])["Amount"].sum().reset_index()
            )
            monthly_estimate = (
                monthly_totals.groupby("Details")["Amount"].mean().reset_index(name="Monthly Estimate")
            )

            summary = summary.merge(monthly_estimate, on="Details", how="left")

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True,
                column_config={
                    "Total": st.column_config.NumberColumn("Total", format="%.2f CAD"),
                    "Average": st.column_config.NumberColumn("Average", format="%.2f CAD"),
                    "Monthly Estimate": st.column_config.NumberColumn("Monthly Estimate", format="%.2f CAD"),
                },
            )

        st.divider()

        st.subheader("Auto-detected recurring candidates")

        tmp = base_df.copy()
        tmp["Month"] = tmp["Date"].dt.to_period("M").astype(str)

        candidates = (
            tmp.groupby("Details")
            .agg(
                Months=("Month", "nunique"),
                Occurrences=("Details", "size"),
                Total=("Amount", "sum"),
                Avg=("Amount", "mean"),
            )
            .reset_index()
            .sort_values(["Months", "Total"], ascending=[False, False])
        )

        min_months = st.slider("Minimum distinct months to consider recurring", 2, 12, 3)
        candidates = candidates[candidates["Months"] >= min_months].head(50)

        st.dataframe(
            candidates,
            width="stretch",
            hide_index=True,
            column_config={
                "Total": st.column_config.NumberColumn("Total", format="%.2f CAD"),
                "Avg": st.column_config.NumberColumn("Avg", format="%.2f CAD"),
            },
        )

        st.caption("Tip: Copy a merchant name (or a short part of it) from this table into the keyword box above.")

    with tab_report:
        st.subheader("Monthly report")

        months = sorted(df["Date"].dt.to_period("M").astype(str).unique(), reverse=True)
        month = st.selectbox("Month", options=months)
        summary = build_month_summary(
            df, month, budgets=st.session_state.budgets, recurring=detected_recurring
        )

        colM1, colM2 = st.columns(2)
        with colM1:
            st.metric("Total spent", f"{summary['total_spent']:,.2f} CAD")
        with colM2:
            st.metric("Transactions", summary["transaction_count"])

        with st.expander("Figures sent to the AI (aggregates only — never transactions)"):
            st.json(summary)

        if not ai_available():
            st.info(
                "AI is not configured — add ANTHROPIC_API_KEY to a .env file to "
                "generate the written summary."
            )
        else:
            if st.button("Generate written summary", type="primary"):
                with st.spinner("Writing the summary…"):
                    try:
                        narrative = generate_narrative(summary)
                        st.session_state[f"narrative_{month}"] = {
                            "text": narrative.text,
                            "suspects": narrative.suspect_numbers,
                        }
                    except NarrativeError as exc:
                        st.error(str(exc))
            if stored := st.session_state.get(f"narrative_{month}"):
                # Streamlit reads paired $ as LaTeX math, so escape the dollar
                # signs the model writes in amounts
                st.markdown(stored["text"].replace("$", r"\$"))
                if stored["suspects"]:
                    st.warning(
                        "Numbers not found in the source figures (possible hallucination): "
                        + ", ".join(stored["suspects"])
                    )
                else:
                    st.caption("✓ Every number in this summary was verified against the source figures.")

    with tab_ask:
        st.subheader("Ask about your transactions")
        st.caption(
            "Your question is translated into a validated filter (never code) and "
            "computed locally with pandas. The filters above do not apply here."
        )
        question = st.text_input(
            "Question",
            placeholder='e.g. "How much did I spend on restaurants since January?"',
            key="ask_question",
        )
        if st.button("Ask", type="primary"):
            if not ai_available():
                st.session_state.ask_error = (
                    "AI is not configured — add ANTHROPIC_API_KEY to a .env file."
                )
                st.session_state.ask_spec = None
            else:
                with st.spinner("Interpreting your question…"):
                    try:
                        spec = translate_question(question)
                        st.session_state.ask_spec = spec.model_dump_json()
                        st.session_state.ask_error = None
                    except QueryTranslationError as exc:
                        st.session_state.ask_error = str(exc)
                        st.session_state.ask_spec = None

        if error := st.session_state.get("ask_error"):
            st.error(error)
        elif spec_json := st.session_state.get("ask_spec"):
            spec = FilterSpec.model_validate_json(spec_json)
            result = execute_spec(spec, df)
            st.caption(f"Interpreted as: {result.description}")

            if result.filtered.empty:
                st.info("No transactions match this filter.")
            elif result.table is not None:
                x_col, y_col = result.table.columns[0], result.table.columns[1]
                fig = px.bar(result.table, x=x_col, y=y_col, title=result.description)
                st.plotly_chart(fig, width="stretch")
                st.dataframe(
                    result.table,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        y_col: st.column_config.NumberColumn(
                            y_col, format="%d" if spec.aggregate == "count" else "%.2f CAD"
                        )
                    },
                )
            else:
                if spec.aggregate == "count":
                    st.metric("Result", f"{int(result.value)}")
                else:
                    st.metric("Result", f"{result.value:,.2f} CAD")
                with st.expander(f"Matching transactions ({len(result.filtered)})"):
                    st.dataframe(
                        result.filtered[["Date", "Details", "Amount", "Category"]],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                            "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD"),
                        },
                    )


if __name__ == "__main__":
    main()
