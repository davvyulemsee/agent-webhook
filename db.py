import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set. Add it in Railway project settings.")

# Connect to Postgres
conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
cursor = conn.cursor()


# Conversations table
cursor.execute("""CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY ,
    customer_phone TEXT,
    message TEXT,
    role TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

# Tickets table
cursor.execute("""CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT,
    urgency TEXT,
    customer_phone TEXT,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

conn.commit()