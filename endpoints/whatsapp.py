from fastapi import APIRouter, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
from models.chat import ChatRequest
from endpoints.chat import chat_endpoint
import logging

router = APIRouter()

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    # Parse the form data sent by Twilio (from the WhatsApp message)
    form_data = await request.form()
    incoming_msg = form_data.get("Body")
    sender = form_data.get("From")  # e.g., "whatsapp:+919588703587"
    session_id = sender  # Use sender's number as session id
    
    # Create a ChatRequest instance using the incoming message.
    chat_request = ChatRequest(input=incoming_msg)
    
    # Call the chat endpoint. Notice that we pass the request object as well.
    chat_response = await chat_endpoint(request, session_id, chat_request)
    response_text = chat_response.response

    # Build Twilio's MessagingResponse and send it back.
    resp = MessagingResponse()
    resp.message(response_text)
    return Response(content=str(resp), media_type="application/xml")
