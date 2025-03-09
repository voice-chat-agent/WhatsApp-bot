# endpoints/whatsapp.py
from fastapi import APIRouter, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from models.chat import ChatRequest
from endpoints.chat import chat_endpoint  # This is your main chat logic
import logging

router = APIRouter()

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    # Parse form data sent by Twilio
    form_data = await request.form()
    incoming_msg = form_data.get("Body")
    sender = form_data.get("From")  # e.g., "whatsapp:+919588703587"
    session_id = sender  # Using the sender's number as the session id

    # Create a ChatRequest using the incoming message
    chat_request = ChatRequest(input=incoming_msg, chat_history=[])
    
    # Call the main chat logic to process the message
    chat_response = await chat_endpoint(session_id, chat_request)
    response_text = chat_response.response

    # Build the Twilio MessagingResponse
    resp = MessagingResponse()
    resp.message(response_text)
    
    # Return the response in Twilio's XML format
    return Response(content=str(resp), media_type="application/xml")
