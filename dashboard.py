import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from datetime import datetime
import time

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0A0A0A, #1A1A1A);
        color: #00FF00;
    }
    .stMetric {
        color: #00FF00;
    }
    .stButton>button {
        background-color: #0A0A0A;
        color: #00FF00;
        border: 1px solid #00FF00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="Amani AI Dashboard",
    layout="wide",   # 🔑 makes the app use full browser width
    initial_sidebar_state="collapsed"
)


st_autorefresh(interval=5000, key="refresh")

# Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL not set. Please add it in Railway environment variables.")
    st.stop()

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

st.title("Amani AI Admin Dashboard")


# Conversations
convos = pd.read_sql("SELECT * FROM conversations ORDER BY timestamp DESC", engine)
tickets = pd.read_sql("SELECT * FROM tickets ORDER BY created_at DESC", engine)


# Layout: 3 panels (25%, 50%, 25%)
col1, col2, col3 = st.columns([1, 2, 1])

# Left Panel: Clients
with col1:
    st.subheader("Clients")
    clients = convos["customer_phone"].unique()
    selected_client = st.selectbox("Select client", clients)

# Middle Panel: Chat History
with col2:
    st.subheader("Chat History")
    if selected_client:
        client_chats = convos[convos["customer_phone"] == selected_client]
        for _, row in client_chats.iterrows():
            # Convert timestamp to HH:MM format
            time_str = pd.to_datetime(row["timestamp"]).strftime("%H:%M")

            role = "🟢 User" if row["role"] == "user" else "🤖 AI"
            st.write(f"[{row['timestamp']}] {role}: {row['message']}")


# Right Panel: Analytics + Tickets
with col3:
    st.subheader("Analytics")
    st.metric("Total Conversations", len(convos))
    st.metric("Escalations", len(tickets))
    rate = round((len(tickets) / len(convos)) * 100, 2) if len(convos) else 0
    st.metric("Escalation Rate (%)", rate)

    # Graphs
    st.write("### Conversations per Client")
    fig1 = px.bar(convos.groupby("customer_phone").size().reset_index(name="count"),
                  x="customer_phone", y="count", title="Conversations per Client")
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### Ticket Status Distribution")
    if "status" in tickets.columns:
        fig2 = px.pie(tickets, names="status", title="Ticket Status Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.write("### Tickets")
    st.dataframe(tickets)
