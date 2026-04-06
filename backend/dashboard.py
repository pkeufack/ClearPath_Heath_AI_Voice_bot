import os
import sqlite3
import pandas as pd
import streamlit as st


st.set_page_config(page_title="ClearPath Call Dashboard", layout="wide")
st.title("ClearPath Call Dashboard")

DB_PATH = os.path.join(os.path.dirname(__file__), "calls.db")


def load_calls_from_db() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "caller_name",
                "caller_phone",
                "transcript",
                "category",
                "action_taken",
                "language",
                "booking_status",
            ]
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
            SELECT
                id,
                timestamp,
                caller_name,
                caller_phone,
                transcript,
                category,
                action_taken,
                language,
                booking_status
            FROM call_logs
            ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "caller_name",
                "caller_phone",
                "transcript",
                "category",
                "action_taken",
                "language",
                "booking_status",
            ]
        )
    finally:
        conn.close()


df = load_calls_from_db()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Calls", int(len(df)))
col2.metric("Category 1", int((df["category"].astype(str) == "1").sum()) if not df.empty else 0)
col3.metric("Category 2", int((df["category"].astype(str) == "2").sum()) if not df.empty else 0)
col4.metric("Category 3", int((df["category"].astype(str) == "3").sum()) if not df.empty else 0)

category_options = ["All", "1", "2", "3"]
selected_category = st.selectbox("Filter by Category", category_options)

filtered_df = df
if selected_category != "All":
    filtered_df = df[df["category"].astype(str) == selected_category]

st.subheader("Call Logs")
if filtered_df.empty:
    st.info("No call records found for the selected filter.")
else:
    st.dataframe(filtered_df, use_container_width=True)
