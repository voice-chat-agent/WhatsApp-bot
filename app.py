# app.py
from fastapi import FastAPI
from endpoints import chat, test_outbound ,whatsapp
from config import logging_config  # Ensure logging is configured

app = FastAPI()

app.include_router(chat.router)
app.include_router(whatsapp.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
