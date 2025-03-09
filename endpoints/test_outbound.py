# endpoints/test_outbound.py
from fastapi import APIRouter
from services.twilio_service import send_whatsapp_template_message

router = APIRouter()

@router.get("/send_test_message")
async def send_test_message():
    recipient = "whatsapp:+919588703587"
    # This is a fallback plain text version of your appointment reminder
    template_text = ""
    sid = send_whatsapp_template_message(recipient, template_text)
    return {"message_sid": sid}
