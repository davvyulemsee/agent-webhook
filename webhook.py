from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import Field
from typing import TypedDict, Sequence, Annotated, Optional
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from anthropic import Anthropic
from datetime import datetime, timedelta
import os
import time
import requests as http_requests

from db import conn, cursor
from admin_handler import handle_admin_command, send_to_admin

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("No GROQ_API_KEY found.")

model = ChatGroq(model="openai/gpt-4o-mini", temperature=0, api_key=groq_api_key)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
ADMIN_PHONE = os.getenv("ADMIN_PHONE")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def search_products(
    query: str = Field(..., description="Search keywords"),
    category: Optional[str] = Field(None, description="Optional category"),
    limit: int = Field(5, description="Number of results"),
) -> str:
    """Search for products or services."""
    params = {"q": query, "limit": min(limit, 10)}
    if category:
        params["category"] = category
    try:
        resp = http_requests.get(f"{API_BASE}/catalog/api/search/", params=params, timeout=5)
        data = resp.json()
        if not data.get("content"):
            return "No results found. Try different keywords."
        return str(data)
    except Exception as e:
        return f"Search error: {e}"


@tool
def escalate_to_human(
    reason: str = Field(..., description="Reason for escalation"),
    urgency: str = Field("medium", description="low / medium / high"),
    customer_phone_or_account: str = Field(..., description="Customer phone number"),
) -> str:
    """Escalate conversation to a human agent."""
    ticket_id = f"TICKET-{int(time.time())}"
    cursor.execute(
        "INSERT INTO tickets (reason, urgency, customer_phone) VALUES (%s, %s, %s)",
        (reason, urgency, customer_phone_or_account)
    )
    conn.commit()
    send_to_admin(
        f"ESCALATION {ticket_id}\n"
        f"Customer: {customer_phone_or_account}\n"
        f"Urgency: {urgency.upper()}\n"
        f"Reason: {reason}"
    )
    return (
        f"I've escalated your query to our team (ticket {ticket_id}). "
        f"Someone will be in touch shortly. Thank you for your patience."
    )


@tool
def book_appointment(
    customer_name: str = Field(..., description="Client's full name"),
    appointment_type: str = Field(..., description="e.g. initial consultation, site visit, design review"),
    preferred_date: str = Field(..., description="Date the client prefers"),
    preferred_time: str = Field(..., description="Time preference"),
    notes: str = Field("", description="Any extra details"),
    customer_phone: str = Field(..., description="Client's phone number"),
) -> str:
    """Book an appointment request."""
    cursor.execute(
        """INSERT INTO appointments
           (customer_phone, customer_name, appointment_type, preferred_date, preferred_time, notes)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (customer_phone, customer_name, appointment_type, preferred_date, preferred_time, notes)
    )
    conn.commit()
    send_to_admin(
        f"NEW APPOINTMENT REQUEST\n"
        f"Client: {customer_name} ({customer_phone})\n"
        f"Type: {appointment_type}\n"
        f"Date: {preferred_date} at {preferred_time}\n"
        f"Notes: {notes or 'None'}"
    )
    return (
        f"Perfect, {customer_name}! Your {appointment_type} request for "
        f"{preferred_date} around {preferred_time} has been logged. "
        f"We will confirm shortly via WhatsApp."
    )


@tool
def search_firm_knowledge(
    query: str = Field(..., description="What the client is asking about"),
) -> str:
    """Search the firm's knowledge base for relevant information."""
    try:
        from rag import search_knowledge_base
        context = search_knowledge_base(query)
        if not context:
            return "No specific information found in our knowledge base."
        return f"Relevant information from our firm:\n\n{context}"
    except Exception as e:
        return f"Knowledge base unavailable: {e}"


tools_list = [book_appointment, escalate_to_human, search_firm_knowledge]


# ── Agent graph ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional, warm AI assistant for an architecture firm.

You help clients with:
- Booking consultations and site visits
- Understanding the firm's design process and services
- Answering questions about fees and project types
- Providing project status updates
- General enquiries about architecture and planning

Rules:
- Always be polite and professional
- Use the search_firm_knowledge tool to answer questions about services, fees, or process
- Use book_appointment when a client wants to schedule a meeting or site visit
- For complex issues or upset clients, use escalate_to_human
- Ask for the client's name early in the conversation if you don't have it
- Keep responses concise — this is WhatsApp, not email
"""),
    MessagesPlaceholder("messages"),
])

llm = prompt | model.bind_tools(tools_list)


def safe_invoke(messages):
    fixed = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            if msg.type == "human":
                fixed.append(HumanMessage(content=msg.content, additional_kwargs=msg.additional_kwargs))
            elif msg.type == "ai":
                fixed.append(AIMessage(content=msg.content, additional_kwargs=msg.additional_kwargs))
            elif msg.type == "system":
                fixed.append(SystemMessage(content=msg.content))
            elif msg.type == "tool":
                fixed.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id))
            else:
                fixed.append(HumanMessage(content=str(msg.content)))
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            if role in ("user", "human"):
                fixed.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                fixed.append(AIMessage(content=content))
            else:
                fixed.append(HumanMessage(content=str(content)))
        else:
            fixed.append(HumanMessage(content=str(msg)))
    return llm.invoke(fixed)


def agent(state: AgentState):
    response = safe_invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("agent_node", agent)
graph.add_node("tools", ToolNode(tools_list))
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", tools_condition)
graph.add_edge("tools", "agent_node")

memory = MemorySaver()
langgraph_app = graph.compile(checkpointer=memory)


# ── Conversation summariser (runs every 15 min) ───────────────────────────────
def summarise_inactive_conversations():
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=30)
        cursor.execute("""
            SELECT DISTINCT customer_phone FROM conversations
            WHERE timestamp < %s
            AND customer_phone NOT IN (
                SELECT DISTINCT customer_phone FROM conversation_summaries
                WHERE created_at > NOW() - INTERVAL '1 hour'
            )
        """, (cutoff,))
        phones = [r["customer_phone"] for r in cursor.fetchall()]

        for phone in phones:
            if phone == ADMIN_PHONE:
                continue
            cursor.execute("""
                SELECT role, message FROM conversations
                WHERE customer_phone = %s
                ORDER BY timestamp ASC
            """, (phone,))
            messages = cursor.fetchall()
            if len(messages) < 3:
                continue

            transcript = "\n".join([f"{r['role'].upper()}: {r['message']}" for r in messages])
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarise this WhatsApp conversation in 2-3 sentences. "
                        f"Focus on what the client wanted and what was resolved:\n\n{transcript}"
                    )
                }]
            )
            summary = response.content[0].text
            cursor.execute(
                """INSERT INTO conversation_summaries (customer_phone, summary)
                   VALUES (%s, %s)
                   ON CONFLICT (customer_phone)
                   DO UPDATE SET summary = %s, created_at = NOW()""",
                (phone, summary, summary)
            )
            conn.commit()
    except Exception as e:
        print(f"Summariser error: {e}")


scheduler = BackgroundScheduler()
scheduler.add_job(summarise_inactive_conversations, "interval", minutes=15)
scheduler.start()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Amani AI Webhook")


@app.get("/")
async def health():
    return {"status": "healthy"}


@app.get("/conversations")
async def get_conversations():
    cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC")
    return JSONResponse(content=[dict(r) for r in cursor.fetchall()])


@app.get("/tickets")
async def get_tickets():
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    return JSONResponse(content=[dict(r) for r in cursor.fetchall()])


@app.get("/analytics")
async def get_analytics():
    cursor.execute("SELECT COUNT(*) as count FROM conversations")
    total_convos = cursor.fetchone()["count"]
    cursor.execute("SELECT COUNT(*) as count FROM tickets")
    total_tickets = cursor.fetchone()["count"]
    return {
        "total_conversations": total_convos,
        "total_tickets": total_tickets,
        "escalation_rate": round((total_tickets / total_convos) * 100, 2) if total_convos else 0,
    }


@app.post("/handoff/{phone}")
async def set_handoff(phone: str, active: bool):
    cursor.execute(
        "UPDATE conversations SET handoff_active = %s WHERE customer_phone = %s",
        (active, phone)
    )
    conn.commit()
    return {"status": "ok", "handoff_active": active}


@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    customer_phone = From.replace("whatsapp:", "").replace("+", "")
    thread_id = f"wa_{customer_phone}"
    config = {"configurable": {"thread_id": thread_id}}

    # ── Admin command mode ────────────────────────────────────────
    if customer_phone == ADMIN_PHONE:
        if Body.strip().startswith("/"):
            reply = handle_admin_command(Body.strip())
        else:
            reply = "Send /help to see available commands."
        resp = MessagingResponse()
        resp.message(reply)
        return Response(content=str(resp), media_type="application/xml")

    # ── Client mode ───────────────────────────────────────────────
    # Log incoming message
    cursor.execute(
        "INSERT INTO conversations (customer_phone, message, role) VALUES (%s, %s, %s)",
        (customer_phone, Body, "user")
    )
    conn.commit()

    # Check if human has taken over
    cursor.execute(
        "SELECT handoff_active FROM conversations WHERE customer_phone = %s "
        "ORDER BY timestamp DESC LIMIT 1",
        (customer_phone,)
    )
    row = cursor.fetchone()
    handoff_active = row["handoff_active"] if row else False

    if handoff_active:
        send_to_admin(
            f"Message from {customer_phone} (handoff active):\n{Body}\n\n"
            f"Reply with: /reply {customer_phone} [your message]"
        )
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="application/xml")

    # AI response
    try:
        inputs = {
            "messages": [
                SystemMessage(content=f"The customer's phone number is {customer_phone}."),
                HumanMessage(content=Body)
            ]
        }
        output = langgraph_app.invoke(inputs, config)
        reply = output["messages"][-1].content if output.get("messages") else "Sorry, I couldn't process that."
    except Exception as e:
        print(f"Agent error: {e}")
        reply = "Sorry, something went wrong. Please try again."

    cursor.execute(
        "INSERT INTO conversations (customer_phone, message, role) VALUES (%s, %s, %s)",
        (customer_phone, reply, "agent")
    )
    conn.commit()

    resp = MessagingResponse()
    resp.message(reply)
    return Response(content=str(resp), media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)