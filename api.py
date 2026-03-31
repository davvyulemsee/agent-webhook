from fastapi import FastAPI
from fastapi.responses import JSONResponse
from db import conn, cursor

app = FastAPI(title="Amani AI Backend API")

@app.get("/conversations")
async def get_conversations():
    cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    return JSONResponse(rows)

@app.get("/tickets")
async def get_tickets():
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return JSONResponse(rows)

@app.get("/analytics")
async def get_analytics():
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total_convos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets")
    total_tickets = cursor.fetchone()[0]

    return {
        "total_conversations": total_convos,
        "total_tickets": total_tickets,
        "escalation_rate": round((total_tickets / total_convos) * 100, 2) if total_convos else 0
    }