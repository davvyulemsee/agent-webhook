import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_document(filename: str, text: str):
    """Chunk a document and store embeddings in Postgres."""
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()

    conn = get_conn()
    cursor = conn.cursor()

    # Remove old version of this file if re-uploading
    cursor.execute("DELETE FROM knowledge_base WHERE filename = %s", (filename,))

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        cursor.execute(
            """INSERT INTO knowledge_base (filename, chunk_index, content, embedding)
               VALUES (%s, %s, %s, %s)""",
            (filename, i, chunk, embedding)
        )
    conn.commit()
    cursor.close()
    conn.close()
    return len(chunks)


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Find the most relevant chunks for a query."""
    query_embedding = model.encode([query])[0].tolist()

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT content, filename,
           1 - (embedding <=> %s::vector) AS similarity
           FROM knowledge_base
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (query_embedding, query_embedding, top_k)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return ""

    context = "\n\n".join([
        f"[From {r['filename']}]:\n{r['content']}"
        for r in results
    ])
    return context