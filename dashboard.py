import streamlit as st
import sqlite3
import pandas as pd

conn = sqlite3.connect("agent_data.db")

st.title("Amani AI Admin Dashboard")

# Conversations
st.header("Conversations")
convos = pd.read_sql("SELECT * FROM conversations ORDER BY timestamp DESC", conn)
st.dataframe(convos)

# Tickets
st.header("Escalation Tickets")
tickets = pd.read_sql("SELECT * FROM tickets ORDER BY created_at DESC", conn)
st.dataframe(tickets)

# Analytics
st.header("Analytics")
st.metric("Total Conversations", len(convos))
st.metric("Escalations", len(tickets))
rate = round((len(tickets) / len(convos)) * 100, 2) if len(convos) else 0
st.metric("Escalation Rate (%)", rate)