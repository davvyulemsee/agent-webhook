from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage
import os
import asyncio

from langchain_groq import ChatGroq
import os
from typing import TypedDict, List, Union, Annotated, Sequence, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import chainlit as cl
from pydantic import Field
import time
import requests
from db import conn, cursor
from pathlib import Path
from fastapi.responses import Response, JSONResponse  # add JSONResponse here

from apscheduler.schedulers.background import BackgroundScheduler
from anthropic import Anthropic
from datetime import datetime, timedelta



from dotenv import load_dotenv

# ================== COPY YOUR FULL AGENT CODE HERE ==================
# Paste the entire content of your virtual_assistant.py (except the Chainlit UI part) below this line

# --- Start of your agent code ---

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("No GROQ_API_KEY! Check .env path.")

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=groq_api_key)

API_BASE = "http://localhost:8000/catalog"

# get bank info ---> RAG to policies. Add later.



anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def summarise_inactive_conversations():
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
                "content": f"Summarise this WhatsApp conversation in 2-3 sentences. Focus on what the client wanted and what was resolved:\n\n{transcript}"
            }]
        )
        summary = response.content[0].text

        cursor.execute(
            "INSERT INTO conversation_summaries (customer_phone, summary) VALUES (%s, %s) "
            "ON CONFLICT (customer_phone) DO UPDATE SET summary = %s, created_at = NOW()",
            (phone, summary, summary)
        )
        conn.commit()

scheduler = BackgroundScheduler()
scheduler.add_job(summarise_inactive_conversations, 'interval', minutes=15)
scheduler.start()


@tool
def book_appointment(
    customer_name: str = Field(..., description="Client's full name"),
    appointment_type: str = Field(..., description="e.g. 'initial consultation', 'site visit', 'design review'"),
    preferred_date: str = Field(..., description="Date the client prefers, e.g. 'Monday 14 April'"),
    preferred_time: str = Field(..., description="Time preference, e.g. '10am' or 'afternoon'"),
    notes: str = Field("", description="Any extra details the client mentioned"),
    customer_phone: str = Field(..., description="Client's phone number")
) -> str:
    """Book an appointment request for an architectural consultation or site visit."""
    cursor.execute(
        """INSERT INTO appointments
           (customer_phone, customer_name, appointment_type, preferred_date, preferred_time, notes)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (customer_phone, customer_name, appointment_type, preferred_date, preferred_time, notes)
    )
    conn.commit()
    return (
        f"Perfect, {customer_name}! I've logged your request for a {appointment_type} "
        f"on {preferred_date} around {preferred_time}. Our team will confirm shortly via WhatsApp."
    )



@tool
def search_products(
        query: str = Field(..., description = "Main search keywords(name, description, category)"),
        category: Optional[str] = Field(None, description="Optional category"),
        limit: int = Field(5, description = "Number of results to show."),
) -> str:
    """Search tool for searching products in store"""
    params = {"q":query, "limit": min(limit, 10)}

    if category:
        params["category"] = category

    try:
        resp = requests.get(f"{API_BASE}/api/search/", params = params, timeout = 5)
        data = resp.json()
        if not data.get("content"):
            return " No products found. Try another product."

        return str(data)
        # return f"Status: {resp.status_code}\nContent-Type: {resp.headers.get('Content-Type')}\nRaw body: {resp.text}"
    except Exception as e:
        return f"Error: {e}"


@tool
def escalate_to_human(
        reason: str = Field(...,
                            description="Brief reason for escalation (e.g., fraud suspicion, complex dispute, customer upset)"),
        urgency: str = Field("medium", description="low / medium / high"),
        customer_phone_or_account: str = Field(..., description="Customer's phone number or account reference")
) -> str:
    """Escalate this conversation to a live human banking agent."""
    ticket_id = f"BANK-{int(time.time())}"
    summary = f"""
            ESCALATION TICKET #{ticket_id}
            URGENCY: {urgency.upper()}
            REASON: {reason}
            CUSTOMER: {customer_phone_or_account}
            CHAT CONTEXT: High-value banking inquiry requiring human review.
            """
    # In demo: just log to console (later → Slack / email / Zendesk API)
    print("=== ESCALATION SENT TO HUMAN TEAM ===\n" + summary)

    return (
    "I'm sorry for the inconvenience. "
    f"I have created ticket #{ticket_id} and escalated it to our customer service team because of: {reason}. "
    f"They will review it promptly based on the {urgency} urgency level. "
    "You will hear back from us soon (usually within 1-2 hours for high urgency). "
    "Thank you for your patience — we will handle this properly."
)


@tool
def check_account_balance(
    account_type: str = Field(..., description="e.g. 'savings', 'current', 'mobile wallet'"),
    last4: str = Field(None, description="Last 4 digits of account/card (optional)")
) -> str:
    """Simulate balance inquiry."""
    return f"{account_type.capitalize()} Account Balance: KSh 48,750.25 (as of today). Available: KSh 45,200. No pending transactions."

@tool
def report_transaction_dispute(
    transaction_id: str = Field(..., description="Transaction reference or date"),
    amount: str = Field(..., description="Amount involved"),
    description: str = Field(..., description="What went wrong")
) -> str:
    """Log dispute and create ticket."""
    return f"Dispute logged for transaction {transaction_id} ({amount}). Ticket #BANK-{int(time.time())} raised. Investigation within 7 days. Pole sana for the inconvenience."

@tool
def check_loan_status(
    loan_reference: str = Field(..., description="Loan account number or ID")
) -> str:
    """Simulate loan status check."""
    return f"Loan {loan_reference}: Outstanding balance KSh 320,000. Next repayment KSh 28,500 due on 10 March 2026. No arrears."

# @tool
# def generate_voice(
#         text: str = Field(..., description="The text response to convert to speech")
# ) -> str:
#     """Convert the agent's text response to spoken audio using ElevenLabs."""
#     try:
#         client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
#         audio = client.text_to_speech.convert(
#             text=text
#             )
#         )
#
#         audio_path = "response."




tools_list = [check_account_balance, report_transaction_dispute, check_loan_status, escalate_to_human, search_products, book_appointment]



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aria, a warm and professional AI assistant for an architectural firm.

        You help clients with:
        - Booking consultations and site visits
        - Understanding the firm's design process and services
        - Answering questions about fees and project types
        - Providing project status updates
        - General enquiries about architecture and planning

        When a new client messages:
        1. Greet them warmly and ask for their name
        2. Ask what kind of project they have in mind — residential, commercial, renovation, interior design, landscape, or something else
        3. Based on their answer, tailor your responses to that project type
        4. If they are ready to move forward, use book_appointment to schedule a consultation

        Project type context:
        - Residential — new homes, extensions, alterations
        - Commercial — offices, retail, hospitality, industrial
        - Renovation — refurbishment of existing structures
        - Interior design — space planning, finishes, furniture layout
        - Landscape — outdoor spaces, gardens, site planning
        - Mixed use — combination of the above

        Rules:
        - Always be polite, warm and professional
        - Never skip asking for the client's name and project type early in the conversation
        - Use the search_firm_knowledge tool to answer questions about services, fees, or process
        - Use book_appointment when a client wants to schedule a meeting or site visit
        - For complex issues or upset clients, use escalate_to_human
        - Keep responses concise — this is WhatsApp, not email
        - Never quote specific prices unless the knowledge base confirms them
        """),
    MessagesPlaceholder("messages"),
])

llm = prompt | model.bind_tools(tools_list)

# - Keep answers clear, short, and in natural Kenyan English


def safe_invoke(messages):
    """
    Force-rebuild every message to make sure we never pass legacy-style HumanMessage/AIMessage
    """
    fixed = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            # Re-create from scratch to avoid legacy constructor issues
            if msg.type == "human":
                fixed.append(HumanMessage(content=msg.content, additional_kwargs=msg.additional_kwargs))
            elif msg.type == "ai":
                fixed.append(AIMessage(content=msg.content, additional_kwargs=msg.additional_kwargs))
            elif msg.type == "system":
                fixed.append(SystemMessage(content=msg.content, additional_kwargs=msg.additional_kwargs))
            elif msg.type == "tool":
                fixed.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id))
            else:
                # fallback
                fixed.append(HumanMessage(content=str(msg.content)))
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            if role in ("user", "human"):
                fixed.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                fixed.append(AIMessage(content=content))
            else:
                fixed.append(HumanMessage(content=str(content)))
        elif isinstance(msg, str):
            fixed.append(HumanMessage(content=msg))
        else:
            fixed.append(HumanMessage(content=str(msg)))

    # Now invoke with the cleaned list
    return llm.invoke(fixed)


def agent(state: AgentState):
    response = safe_invoke(state['messages'])
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("agent_node", agent)

tools = ToolNode(tools_list)

graph.add_node("tools", tools)

graph.add_edge(START, "agent_node")

graph.add_conditional_edges("agent_node", tools_condition)

graph.add_edge("tools", "agent_node")


memory = MemorySaver()

# app = graph.compile(checkpointer=memory)
langgraph_app = graph.compile(checkpointer=memory)


# ================== END OF AGENT CODE ==================

app = FastAPI(title="Amani AI WhatsApp Webhook")

@app.get("/conversations")
async def get_conversations():
    cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    return JSONResponse(content=[dict(r) for r in rows])

@app.get("/tickets")
async def get_tickets():
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return JSONResponse(content=[dict(r) for r in rows])

@app.get("/analytics")
async def get_analytics():
    cursor.execute("SELECT COUNT(*) as count FROM conversations")
    total_convos = cursor.fetchone()["count"]
    cursor.execute("SELECT COUNT(*) as count FROM tickets")
    total_tickets = cursor.fetchone()["count"]
    return {
        "total_conversations": total_convos,
        "total_tickets": total_tickets,
        "escalation_rate": round((total_tickets / total_convos) * 100, 2) if total_convos else 0
    }



@app.get("/")
async def health():
    return {"status": "healthy", "message": "Amani WhatsApp webhook is running"}

@app.get("/test-db")
async def test_db():
    cursor.execute("SELECT NOW()")
    result = cursor.fetchone()
    return {"db_time": result["now"]}

@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    thread_id = f"wa_{From.replace('whatsapp:', '').replace('+', '')}"
    config = {"configurable": {"thread_id": thread_id}}
    customer_phone = From.replace("whatsapp:", "").replace("+", "")

    # Log user message
    cursor.execute(
        "INSERT INTO conversations (customer_phone, message, role) VALUES (%s, %s, %s)",
        (customer_phone, Body, "user")
    )
    conn.commit()

    # Check if human has taken over this conversation
    cursor.execute(
        "SELECT handoff_active FROM conversations WHERE customer_phone = %s "
        "ORDER BY timestamp DESC LIMIT 1",
        (customer_phone,)
    )
    row = cursor.fetchone()
    handoff_active = row["handoff_active"] if row else False

    if handoff_active:
        # Human is handling it — just acknowledge silently, don't invoke AI
        resp = MessagingResponse()
        return Response(content=str(resp), media_type="application/xml")

    # Normal AI flow
    try:
        # In the webhook, before invoking:
        inputs = {
            "messages": [
                SystemMessage(content=f"The customer's phone number is {customer_phone}."),
                HumanMessage(content=Body)
            ]
        }
        output = langgraph_app.invoke(inputs, config)
        reply = output["messages"][-1].content if output.get("messages") else "Sorry, I couldn't process that."
    except Exception:
        reply = "Sorry, something went wrong. Please try again later."

    cursor.execute(
        "INSERT INTO conversations (customer_phone, message, role) VALUES (%s, %s, %s)",
        (customer_phone, reply, "agent")
    )
    conn.commit()

    resp = MessagingResponse()
    resp.message(reply)
    return Response(content=str(resp), media_type="application/xml")

@app.post("/handoff/{phone}")
async def set_handoff(phone: str, active: bool):
    cursor.execute(
        "UPDATE conversations SET handoff_active = %s WHERE customer_phone = %s",
        (active, phone)
    )
    conn.commit()
    return {"status": "ok", "handoff_active": active}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)