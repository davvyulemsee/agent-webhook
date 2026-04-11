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




tools_list = [check_account_balance, report_transaction_dispute, check_loan_status, escalate_to_human, search_products]



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional, calm, and empathetic AI Banking Assistant for Kenyan banks such as Equity, KCB, Co-operative Bank, NCBA, or Absa.

    You assist customers with:
    - Checking account balances and recent transactions
    - Inquiring about transaction disputes or unexpected charges
    - Checking loan status and repayment schedules
    - Mobile or internet banking registration, PIN reset, or app issues
    - Card activation, blocking, or lost/stolen card procedures
    - General banking questions (account opening, fees, branch locations, M-Pesa linked accounts)

    Rules:
    - Always be polite, patient, and understanding
    - Use clear, concise English
    - Use tools for balance checks, loan status, disputes, or policy information — never guess real numbers or details
    - If the question is unclear, ask for clarification (e.g., account type, last 4 digits, transaction reference)
    - For complex or sensitive matters (fraud, large sums, personal data changes, branch visits, or angry customers), use the escalate_to_human tool
    - After escalation, provide the returned message and end the conversation politely
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
        inputs = {"messages": [HumanMessage(content=Body)]}
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