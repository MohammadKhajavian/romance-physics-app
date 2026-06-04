# pages/2_Admin.py
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Admin Dashboard", page_icon="📊")

st.title("📊 Model Performance Dashboard")
st.markdown("This page shows how the ML model is learning from user data.")

DB_PATH = "data/romance_data.db"

if not os.path.exists(DB_PATH):
    st.warning("No database found yet. Be the first user on the main page!")
    st.stop()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM users", conn)
conn.close()

st.metric("Total Data Points", len(df))
st.metric("Success Rate (predictions of new love)", f"{df['prediction_result'].mean() * 100:.1f}%")

if len(df) > 0:
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x='age', title='Age Distribution', color_discrete_sequence=['blue'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='first_love_intensity', title='Past Relationship Intensity',
                            color_discrete_sequence=['red'])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(df.tail(10))

    # Download button
    csv = df.to_csv(index=False)
    st.download_button("Download Data as CSV", csv, "romance_data.csv", "text/csv")
else:
    st.info("No data yet. Use the main prediction page first.")