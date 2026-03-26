from fastapi import FastAPI, Form
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Import your compiled LangGraph app
# Option A: If you want to keep it simple for now, create a minimal version
# Option B: Copy your full agent code here (recommended for first test)

load_dotenv()

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    thread_id = f"wa_{From.replace('whatsapp:', '').replace('+', '')}"

    # For now, we'll use a simple echo to test the connection
    # Later we'll connect your full LangGraph agent
    reply = f"Received: {Body}\n\nI'm Amani AI. I'm still learning, but I'm here to help with banking questions!"

    resp = MessagingResponse()
    resp.message(reply)

    return str(resp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)