import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Simple Finance App", page_icon="💰", layout="wide")

category_file = "categories.json"
budget_file = "budgets.json"
recurring_file = "recurring.json"

if "budgets" not in st.session_state:
    st.session_state.budgets = {}
    if os.path.exists(budget_file):
        try:
            with open(budget_file, "r", encoding="utf-8") as f:
                st.session_state.budgets = json.load(f)
        except Exception:
            st.session_state.budgets = {}

if "categories" not in st.session_state:
    st.session_state.categories = {"Uncategorized": []}
    if os.path.exists(category_file):
        try:
            with open(category_file, "r", encoding="utf-8") as f:
                st.session_state.categories = json.load(f)
        except Exception:
            st.session_state.categories = {"Uncategorized": []}
    st.session_state.categories.setdefault("Uncategorized", [])

if "recurring" not in st.session_state:
    st.session_state.recurring = []
    if os.path.exists(recurring_file):
        try:
            with open(recurring_file, "r", encoding="utf-8") as f:
                st.session_state.recurring = json.load(f)
        except Exception:
            st.session_state.recurring = []


def save_categories():
    with open(category_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.categories, f, ensure_ascii=False)


def save_budgets():
    with open(budget_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.budgets, f, ensure_ascii=False)


def save_recurring():
    with open(recurring_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.recurring, f, ensure_ascii=False)


def categorize_transactions(df):
    df["Category"] = "Uncategorized"
    details_lower = df["Details"].astype(str).str.lower().str.strip()

    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords:
            continue
        lowered_keywords = [str(k).lower().strip() for k in keywords if str(k).strip()]
        if not lowered_keywords:
            continue
        mask = details_lower.apply(lambda s: any(k in s for k in lowered_keywords))
        df.loc[mask, "Category"] = category

    return df


DATE_ALIASES = [
    "transaction date", "date", "posted date", "trade date",
    "date de transaction", "date d'opération", "date operation",
]
DETAILS_ALIASES = [
    "description", "details", "merchant", "payee", "name", "memo",
    "libellé", "référence", "reference",
]
AMOUNT_ALIASES = [
    "transaction amount", "amount", "montant", "valeur", "amount (cad)", "amount cad",
]
DEBIT_ALIASES = ["debit", "withdrawal", "sortie", "débit"]
CREDIT_ALIASES = ["credit", "deposit", "entrée", "crédit"]


def _norm_col(c: str) -> str:
    return str(c).strip().lower()


def _find_col(df, aliases):
    cols = list(df.columns)
    norm = {c: _norm_col(c) for c in cols}

    # exact match
    for a in aliases:
        a = a.lower()
        for c, n in norm.items():
            if n == a:
                return c

    # contains match
    for a in aliases:
        a = a.lower()
        for c, n in norm.items():
            if a in n:
                return c

    return None


def _smart_read_csv(file) -> pd.DataFrame:
    """
    Try several encodings, separators and skiprows values (banks often add
    preamble lines above the header). Picks the parse with the most columns.
    """
    best_df = None
    best_cols = -1

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        decoded_any = False
        for sep in (",", ";", "\t"):
            for skip in range(7):
                try:
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=skip, sep=sep, encoding=encoding)
                except Exception:
                    continue
                decoded_any = True
                if not df.empty and df.shape[1] > best_cols:
                    best_df = df
                    best_cols = df.shape[1]
        if decoded_any:
            # This encoding decoded the bytes fine; trying more would only
            # reinterpret the same data.
            break

    if best_df is None:
        raise ValueError("Could not read CSV (tried multiple encodings, separators and header offsets).")

    best_df.columns = [str(c).strip() for c in best_df.columns]
    return best_df


def _clean_amount(value):
    """Parse one amount string: $1,234.56 / 1 234,56 / (123.45) / -12,50 ..."""
    v = str(value).strip()
    if v.lower() in ("", "nan", "none", "-"):
        return None

    negative = v.startswith("(") and v.endswith(")")
    if negative:
        v = v[1:-1]
    if v.lstrip().startswith("-"):
        negative = True

    # keep only digits and separators (drops $, spaces, nbsp, currency codes)
    v = re.sub(r"[^\d,.]", "", v)
    if not v:
        return None

    if "," in v and "." in v:
        # the rightmost separator is the decimal one
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "").replace(",", ".")
        else:
            v = v.replace(",", "")
    elif "," in v:
        head, _, tail = v.rpartition(",")
        if len(tail) <= 2:  # decimal comma: 12,50
            v = head.replace(",", "") + "." + tail
        else:  # thousands separator: 1,234
            v = v.replace(",", "")

    try:
        n = float(v)
    except ValueError:
        return None
    return -n if negative else n


def _parse_amount_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.map(_clean_amount), errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # Day-first heuristic: if the first component ever exceeds 12 (and the
    # second never does), the format must be DD/MM/YYYY.
    dayfirst = False
    parts = s.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
    if parts[0].notna().any():
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        if (first > 12).any() and not (second > 12).any():
            dayfirst = True

    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")


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


def _normalize_transactions(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame()

    out["Date"] = _parse_dates(df_raw[mapping["date_col"]])
    out["Details"] = df_raw[mapping["details_col"]].astype(str).str.strip()

    if mapping["mode"] == "single":
        amt = _parse_amount_series(df_raw[mapping["amount_col"]])
        out["Debit/Credit"] = amt.apply(lambda x: "Credit" if pd.notna(x) and x < 0 else "Debit")
        out["Amount"] = amt.abs()
    else:
        debit = _parse_amount_series(df_raw[mapping["debit_col"]]).abs().fillna(0)
        credit = _parse_amount_series(df_raw[mapping["credit_col"]]).abs().fillna(0)
        out["Debit/Credit"] = (credit > 0).map({True: "Credit", False: "Debit"})
        total = debit + credit
        out["Amount"] = total.where(total > 0)  # rows with no amount at all become NaN

    out = out.dropna(subset=["Date", "Amount"])
    return out[["Date", "Details", "Amount", "Debit/Credit"]].reset_index(drop=True)


def _mapping_matches(mapping: dict, df: pd.DataFrame) -> bool:
    return all(mapping[k] in df.columns for k in mapping if k.endswith("_col"))


def load_transactions(file):
    try:
        df_raw = _smart_read_csv(file)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    # ---- Attempt auto-detect mapping ----
    date_col = _find_col(df_raw, DATE_ALIASES)
    details_col = _find_col(df_raw, DETAILS_ALIASES)
    amount_col = _find_col(df_raw, AMOUNT_ALIASES)
    debit_col = _find_col(df_raw, DEBIT_ALIASES)
    credit_col = _find_col(df_raw, CREDIT_ALIASES)

    mapping = None

    if date_col and details_col and amount_col:
        mapping = {
            "mode": "single",
            "date_col": date_col,
            "details_col": details_col,
            "amount_col": amount_col,
        }
    elif date_col and details_col and debit_col and credit_col:
        mapping = {
            "mode": "split",
            "date_col": date_col,
            "details_col": details_col,
            "debit_col": debit_col,
            "credit_col": credit_col,
        }

    # ---- Manual mapping fallback (persisted per file) ----
    if mapping is None:
        stored = st.session_state.get("manual_mapping")
        if stored and stored.get("file") == file.name and _mapping_matches(stored["mapping"], df_raw):
            mapping = stored["mapping"]
        else:
            mapping = _mapping_ui(df_raw)
            if mapping is None:
                return None
            st.session_state.manual_mapping = {"file": file.name, "mapping": mapping}

    try:
        df = _normalize_transactions(df_raw, mapping)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    if df.empty:
        st.error("No valid transactions found in this file. Check the column mapping and the file contents.")
        st.session_state.pop("manual_mapping", None)
        return None

    return categorize_transactions(df)


def add_keyword_to_category(category, keyword):
    keyword = str(keyword).strip()
    if not keyword:
        return False

    if category not in st.session_state.categories:
        st.session_state.categories[category] = []

    existing = {k.lower().strip() for k in st.session_state.categories[category]}
    if keyword.lower() in existing:
        return False

    st.session_state.categories[category].append(keyword)
    save_categories()
    return True


def main():
    st.title("Finance Dashboard")

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a bank CSV export to get started.")
        return

    df = load_transactions(uploaded_file)
    if df is None:
        return

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

    tab1, tab2, tab3 = st.tabs(["Expenses (Debits)", "Payments (Credits)", "Recurring"])

    with tab1:
        new_category = st.text_input("New Category Name")
        add_button = st.button("Add Category")

        if add_button and new_category:
            if new_category not in st.session_state.categories:
                st.session_state.categories[new_category] = []
                save_categories()
                st.rerun()

        st.subheader("Your Expenses")
        edited_df = st.data_editor(
            st.session_state.debits_df[["Date", "Details", "Amount", "Category"]],
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD"),
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(st.session_state.categories.keys())
                ),
            },
            hide_index=True,
            width="stretch",
            key="category_editor",
        )
        save_button = st.button("Apply Changes", type="primary")
        if save_button:
            changed = False
            for idx, row in edited_df.iterrows():
                chosen_category = row["Category"]
                if chosen_category == st.session_state.debits_df.at[idx, "Category"]:
                    continue
                st.session_state.debits_df.at[idx, "Category"] = chosen_category
                add_keyword_to_category(chosen_category, row["Details"])
                changed = True
            if changed:
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

        cats = list(st.session_state.categories.keys())
        selected_cat = st.selectbox("Choose a category", options=cats)

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

    with tab3:
        st.subheader("Recurring payments / subscriptions")

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


if __name__ == "__main__":
    main()
