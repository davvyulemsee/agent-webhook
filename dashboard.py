import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL not set. Please add it in Railway environment variables.")
    st.stop()

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

st.title("Amani AI Admin Dashboard")

# Conversations
st.header("Conversations")
convos = pd.read_sql("SELECT * FROM conversations ORDER BY timestamp DESC", engine)
st.dataframe(convos)

# Tickets
st.header("Escalation Tickets")
tickets = pd.read_sql("SELECT * FROM tickets ORDER BY created_at DESC", engine)
st.dataframe(tickets)

# Analytics
st.header("Analytics")
st.metric("Total Conversations", len(convos))
st.metric("Escalations", len(tickets))
rate = round((len(tickets) / len(convos)) * 100, 2) if len(convos) else 0
st.metric("Escalation Rate (%)", rate)