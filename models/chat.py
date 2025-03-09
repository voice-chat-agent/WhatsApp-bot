# models/chat.py
from typing import List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    input: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[dict]
