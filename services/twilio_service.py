# services/twilio_service.py
from twilio.rest import Client
from config.env import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp_template_message(to: str, template_text: str) -> str:
    """
    Send a WhatsApp message using a plain text body as a fallback.
    :param to: Recipient's number in E.164 format with 'whatsapp:' prefix.
    :param template_text: The text content that mimics your template.
    :return: The SID of the sent message.
    """
    message = client.messages.create(
        body=template_text,
        from_=TWILIO_PHONE_NUMBER,  # e.g., "whatsapp:+14155238886"
        to=to                      # e.g., "whatsapp:+919588703587"
    )
    return message.sid
