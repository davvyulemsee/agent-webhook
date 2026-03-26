from fastapi import FastAPI, Form
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# ================== COPY YOUR FULL AGENT CODE HERE ==================
# Paste the entire content of your virtual_assistant.py (except the Chainlit UI part) below this line

# --- Start of your agent code ---

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from pydantic import Field
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import time

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("No GROQ_API_KEY found!")

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

# Your tools
@tool
def check_account_balance(account_type: str, last4: str = None):
    return f"{account_type.capitalize()} Account Balance: KSh 48,750.25. Available: KSh 45,200."

@tool
def report_transaction_dispute(transaction_id: str, amount: str, description: str):
    return f"Dispute logged for transaction {transaction_id} ({amount}). Ticket created. We'll investigate within 7 days."

@tool
def check_loan_status(loan_reference: str):
    return f"Loan {loan_reference}: Outstanding KSh 320,000. Next repayment due 10 March 2026."

@tool
def escalate_to_human(reason: str, urgency: str = "medium", customer_phone_or_account: str = ""):
    ticket_id = f"BANK-{int(time.time())}"
    return f"I'm sorry for the inconvenience. Ticket #{ticket_id} has been created for: {reason}. A human agent will contact you soon."

tools_list = [check_account_balance, report_transaction_dispute, check_loan_status, escalate_to_human]

# System prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional, calm, and empathetic AI Banking Assistant.
You help with account balances, transaction disputes, loan status, and general banking questions.
Use tools when needed. Be polite and clear."""),
    MessagesPlaceholder("messages"),
])

llm = prompt | model.bind_tools(tools_list)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent_node", agent)
graph.add_node("tools", ToolNode(tools_list))
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", tools_condition)
graph.add_edge("tools", "agent_node")

memory = MemorySaver()
langgraph_app = graph.compile(checkpointer=memory)

# ================== END OF AGENT CODE ==================

app = FastAPI(title="Amani AI WhatsApp Webhook")

@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    thread_id = f"wa_{From.replace('whatsapp:', '').replace('+', '')}"

    config = {"configurable": {"thread_id": thread_id}}

    try:
        inputs = {"messages": [HumanMessage(content=Body)]}
        output = langgraph_app.invoke(inputs, config)
        reply = output["messages"][-1].content if output.get("messages") else "Sorry, I couldn't process that. Please try again."
    except Exception as e:
        reply = "Sorry, something went wrong. Please try again later."

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)