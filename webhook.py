from fastapi import FastAPI, Form
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Import your compiled LangGraph app from the main file
# Make sure the filename matches exactly
from ..virtual_assistant.virtual_assistant import app as langgraph_app

load_dotenv()

app = FastAPI(title="Amani AI WhatsApp Webhook")

@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    """
    Receives messages from Twilio WhatsApp and forwards them to your LangGraph agent.
    """
    # Create a unique thread ID for each WhatsApp user
    thread_id = f"wa_{From.replace('whatsapp:', '').replace('+', '')}"

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run your LangGraph agent
        inputs = {"messages": [HumanMessage(content=Body)]}
        output = langgraph_app.invoke(inputs, config)

        # Get the last message from the agent
        reply = output["messages"][-1].content if output.get("messages") else \
                "Sorry, I couldn't process that. Please try again."

    except Exception as e:
        reply = f"Sorry, something went wrong: {str(e)[:100]}"

    # Send reply back to WhatsApp via Twilio
    resp = MessagingResponse()
    resp.message(reply)

    return str(resp)


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)