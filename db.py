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
    id SERIAL PRIMARY KEY ,
    reason TEXT,
    urgency TEXT,
    customer_phone TEXT,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

# appointments
cursor.execute("""CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    customer_phone TEXT,
    customer_name TEXT,
    appointment_type TEXT,
    preferred_date TEXT,
    preferred_time TEXT,
    notes TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        id SERIAL PRIMARY KEY,
        customer_phone TEXT UNIQUE,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id SERIAL PRIMARY KEY,
        customer_phone TEXT,
        customer_name TEXT,
        appointment_type TEXT,
        preferred_date TEXT,
        preferred_time TEXT,
        notes TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Knowledge base table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Vector similarity index
cursor.execute("""
    CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx 
    ON knowledge_base USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
""")

conn.commit()

conn.commit()


conn.commit()