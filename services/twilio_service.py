# services/twilio_service.py
from twilio.rest import Client
from config.env import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp_message(to: str, body: str) -> str:
    """
    Sends a WhatsApp text via Twilio.
    :param to: e.g. "whatsapp:+91XXXXXXXXXX"
    :param body: message text
    :returns: Message SID
    """
    message = client.messages.create(
        from_=TWILIO_PHONE_NUMBER,
        to=to,
        body=body
    )
    return message.sid
