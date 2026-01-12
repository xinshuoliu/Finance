import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="simple finance app", page_icon="💰", layout="wide")

category_file = "categories.json"

budget_file = "budgets.json"

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

if os.path.exists(budget_file):
    try:
        with open(budget_file, "r") as f:
            st.session_state.budgets = json.load(f)
    except Exception:
        st.session_state.budgets = {}


if "categories" not in st.session_state:
    st.session_state.categories = {
        "Uncategorized": []
    }

if os.path.exists(category_file):
    with open(category_file, "r") as f:
        st.session_state.categories = json.load(f)

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def save_budgets():
    with open(budget_file, "w") as f:
        json.dump(st.session_state.budgets, f)

def categorize_transactions(df):
    df["Category"] = "Uncategorized"

    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords:
            continue
        lowered_keywords = [keyword.lower().strip() for keyword in keywords]
        for idx, row in df.iterrows():
            details = row["Details"].lower().strip()
            if any(k in details for k in lowered_keywords):
                df.at[idx, "Category"] = category

    return df

recurring_file = "recurring.json"

if "recurring" not in st.session_state:
    st.session_state.recurring = []

if os.path.exists(recurring_file):
    try:
        with open(recurring_file, "r") as f:
            st.session_state.recurring = json.load(f)
    except Exception:
        st.session_state.recurring = []

def save_recurring():
    with open(recurring_file, "w") as f:
        json.dump(st.session_state.recurring, f)


def load_transactions(file):
    try:
        df = pd.read_csv(file, skiprows=2)

        df.columns = [c.strip() for c in df.columns]

        df["Details"] = df["Description"].astype(str).str.strip()

        df["Amount"] = (
            df["Transaction Amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        df["Debit/Credit"] = df["Amount"].apply(lambda x: "Credit" if x < 0 else "Debit")

        df["Date"] = pd.to_datetime(df["Transaction Date"].astype(str), format="%Y%m%d")

        df["Amount"] = df["Amount"].abs()

        df = df[["Date", "Details", "Amount", "Debit/Credit"]]

        return categorize_transactions(df)

    except Exception as e:
        st.error(f"error processing file: {str(e)}")
        return None
    
def add_keyword_to_category(category, keyword):
    keyword = str(keyword).strip()
    if not keyword:
        return False

    # Make sure the category exists
    if category not in st.session_state.categories:
        st.session_state.categories[category] = []

    # Avoid duplicates (case-insensitive)
    existing = {k.lower().strip() for k in st.session_state.categories[category]}
    if keyword.lower() in existing:
        return False

    st.session_state.categories[category].append(keyword)
    save_categories()
    return True 

def main():
    st.title("Finance Dashboard")

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"]) 

    if uploaded_file is not None:
        df = load_transactions(uploaded_file)

        if df is not None: 
            st.subheader("Filters")

            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                min_date = df["Date"].min().date()
                max_date = df["Date"].max().date()
                start_date, end_date = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

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
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="category_editor"
                )
                save_button = st.button("Apply Changes", type="primary")
                if save_button:
                    for idx, row in edited_df.iterrows():
                        new_category = row["Category"]
                        if new_category == st.session_state.debits_df.at[idx, "Category"]:
                            continue
                        
                        details = row["Details"]
                        st.session_state.debits_df.at[idx, "Category"] = new_category
                        add_keyword_to_category(new_category, details)

                st.subheader('Expense Summary')
                category_totals = st.session_state.debits_df.groupby("Category")["Amount"].sum().reset_index()
                category_totals = category_totals.sort_values("Amount", ascending=False)      

                st.dataframe(category_totals, column_config={
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f CAD")
                }, 
                use_container_width=True,
                hide_index=True
            )  
            
            st.subheader("Budgets (per category)")

            cats = list(st.session_state.categories.keys())
            selected_cat = st.selectbox("Choose a category", options=cats)

            current_budget = float(st.session_state.budgets.get(selected_cat, 0) or 0.0)
            new_budget = st.number_input(
                "Budget (CAD) for the current filtered period",
                min_value=0.0,
                value=current_budget,
                step=10.0
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
                    ratio = spent / budget if budget else 0.0

                    if spent > budget:
                        st.warning(f"⚠️ {cat}: {spent:.2f} CAD spent (budget {budget:.2f} CAD)")
                        st.write(f"Over by: {(spent - budget):.2f} CAD")
                    else:
                        remaining = budget - spent
                        st.success(f"✅ {cat}: {spent:.2f} CAD spent — {remaining:.2f} CAD remaining")
                        st.write(f"Used: {ratio*100:.0f}%")

                    st.progress(min(ratio, 1.0))

            
            fig= px.pie(
                category_totals,
                values="Amount",
                names="Category",
                title="Expenses by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Payment Summary")
                total_payments = credits_df["Amount"].sum()
                st.metric("Total Payments", f"{total_payments:.2f} CAD")
                st.write(credits_df)

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
                            st.success(f"Added: {k}")

                if st.session_state.recurring:
                    colC, colD = st.columns([2, 1])
                    with colC:
                        to_remove = st.selectbox("Remove keyword", options=st.session_state.recurring)
                    with colD:
                        if st.button("Remove selected"):
                            st.session_state.recurring = [x for x in st.session_state.recurring if x != to_remove]
                            save_recurring()
                            st.info(f"Removed: {to_remove}")

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
                        use_container_width=True,
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
                        use_container_width=True,
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
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total": st.column_config.NumberColumn("Total", format="%.2f CAD"),
                        "Avg": st.column_config.NumberColumn("Avg", format="%.2f CAD"),
                    },
                )

                st.caption("Tip: Copy a merchant name (or a short part of it) from this table into the keyword box above.")


main()