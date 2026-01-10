import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="simple finance app", page_icon="💰", layout="wide")

category_file = "categories.json"

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
            debits_df = df[df["Debit/Credit"] == "Debit"].copy()
            credits_df = df[df["Debit/Credit"] == "Credit"].copy()

            st.session_state.debits_df = debits_df.copy()

            tab1, tab2 = st.tabs(["Expenses (Debits)", "Payments (Credits)"])
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

main()