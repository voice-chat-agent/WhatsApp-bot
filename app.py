# # app.py
# from fastapi import FastAPI
# from endpoints import chat, test_outbound ,whatsapp
# from config import logging_config  # Ensure logging is configured

# app = FastAPI()

# app.include_router(chat.router)
# app.include_router(whatsapp.router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# app.py

import os
import re
from uuid import uuid4
from datetime import datetime, time, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from langchain.llms.base import LLM
from db.mongodb import db

# 1. Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GOOGLE_GENAI_API_KEY environment variable.")

# 2. System prompt
SYSTEM_PROMPT = (
    "You are a professional medical assistant chatbot for our hospital. "
    "You will grreetly assist users"
    "You will help users find the right specialist and book an appointment.\n"
    "You will only respond with the specialist name and appointment details.\n"
    "Follow these rules EXACTLY:\n"
    "1. If the user describes symptoms, output **only** “Consult a <Specialist>.”\n"
    "2. If the user directly asks for a specialist, treat it like rule 1.\n"
    "3. Do not echo user text or add any other content.\n"
    "4. After “Consult a X.” we will look up that X in our database and handle booking.\n"
)

# 3. MongoDB collections
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

# 4. Google GenAI LLM wrapper
class GoogleGenAI(LLM):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7

    def __init__(self, model_name=None, temperature=None, api_key=None):
        super().__init__()
        self.model_name = model_name or self.model_name
        self.temperature = temperature or self.temperature
        self._client = genai.Client(api_key=api_key)

    def _call(self, prompt: str, stop=None) -> str:
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return resp.text

    @property
    def _llm_type(self) -> str:
        return "google_genai"

llm = GoogleGenAI(api_key=API_KEY)

# 5. In-memory session store
sessions = {}

# 6. Helpers
def build_prompt(history: list, user_message: str) -> str:
    history.append({"role": "user", "content": user_message})
    prompt = SYSTEM_PROMPT + "\n\n"
    for turn in history:
        tag = "<|user|>" if turn["role"] == "user" else "<|assistant|>"
        prompt += f"{tag}\n{turn['content']}\n\n"
    prompt += "<|assistant|>\n"
    return prompt


def extract_specialty(text: str) -> str:
    lead_ins = [
        r"consult a", r"consult", r"see a", r"see", r"visit a", r"visit",
        r"i am looking for", r"looking for", r"need a", r"search for", r"want a"
    ]
    pattern = re.compile(r"^(?:" + r"|".join(lead_ins) + r")\s+", re.IGNORECASE)
    s = text.strip()
    s = pattern.sub("", s)
    s = s.rstrip(".!?; ")
    return s.title()


def is_valid_slot(dt: datetime) -> bool:
    t = dt.time()
    return time(9, 0) <= t <= time(21, 0) and (t.minute % 10) == 0


def find_next_slot(doctor_id: int, start: datetime) -> datetime:
    """
    Find the next available 10-minute slot between 09:00 and 21:00 on the same or subsequent days.
    """
    slot = start
    # Start at next 10-minute increment
    minute = (slot.minute // 10) * 10
    if slot.minute % 10 != 0:
        minute += 10
    slot = slot.replace(minute=minute, second=0, microsecond=0)

    # Search up to a week ahead
    for _ in range(7 * 12):  # 12 slots per 2-hour block × 7 days approx
        if is_valid_slot(slot):
            exists = appointments_collection.find_one({
                "doctor_id":        doctor_id,
                "appointment_time": slot
            })
            if not exists:
                return slot
        slot += timedelta(minutes=10)
        # If past 21:00, jump to next day at 09:00
        if slot.time() > time(21, 0):
            slot = slot.replace(hour=9, minute=0) + timedelta(days=1)
    raise HTTPException(status_code=404, detail="No available slots found within 7 days.")

# 7. Request/Response models
class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

# 8. FastAPI app & endpoint
app = FastAPI(title="Medical Chat + Booking Agent", version="1.0")

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    state = sessions.setdefault(session_id, {"stage": "start", "history": []})
    user_msg = req.user_message.strip()

    # START stage
    if state["stage"] == "start":
        prompt = build_prompt(state["history"], user_msg)
        suggestion = llm(prompt).strip()
        if not suggestion.lower().startswith("consult a "):
            fallback = (
                SYSTEM_PROMPT
                + "\n\nPlease respond **only** in the form “Consult a <Specialist>.”\n\n"
                f"<|user|>\n{user_msg}\n\n<|assistant|>\n"
            )
            suggestion = llm(fallback).strip()
        state["history"].append({"role": "assistant", "content": suggestion})
        m = re.match(r"^Consult a (.+)\.$", suggestion, re.IGNORECASE)
        if not m:
            return ChatResponse(session_id=session_id, reply="Sorry, I couldn’t identify a specialist.")
        specialty = m.group(1).strip()
        doc = doctors_collection.find_one({
            "$or": [
                {"specialty":         {"$regex": f"^{re.escape(specialty)}$", "$options": "i"}},
                {"clinic_interests":   {"$regex": f"^{re.escape(specialty)}$", "$options": "i"}}
            ]
        })
        if not doc:
            return ChatResponse(session_id=session_id, reply=f"Sorry, we do not have a {specialty} specialist in our hospital.")
        name = doc.get("name", "").title() or "Unknown"
        avail = doc.get("availability", False)
        status = "available" if avail else "not available"
        state.update({
            "stage":  "await_booking",
            "doctor": {"id": doc["doctor_id"], "name": name}
        })
        return ChatResponse(
            session_id=session_id,
            reply=(
                f"{suggestion} We have Dr. {name}, and they are {status}. "
                "Would you like to book an appointment with them? (yes/no)"
            )
        )

    # AWAIT_BOOKING stage
    if state["stage"] == "await_booking":
        if user_msg.lower().startswith("y"):
            state["stage"] = "await_details"
            state["history"].append({"role": "user", "content": user_msg})
            return ChatResponse(session_id=session_id, reply="Please provide your Name, Age, Gender, Contact No, and Symptoms.")
        sessions.pop(session_id, None)
        return ChatResponse(session_id=session_id, reply="Understood. Let me know if you need anything else.")

    # AWAIT_DETAILS stage
    if state["stage"] == "await_details":
        state["patient"] = user_msg
        state["stage"] = "await_slot"
        state["history"].append({"role": "user", "content": user_msg})
        return ChatResponse(
            session_id=session_id,
            reply=(
                "Thank you. Now provide a preferred slot in YYYY‑MM‑DD HH:MM "
                "(between 09:00 and 21:00, every 10 minutes)."
            )
        )

    # AWAIT_SLOT stage
    if state["stage"] == "await_slot":
        try:
            slot_dt = datetime.strptime(user_msg, "%Y-%m-%d %H:%M")
        except ValueError:
            return ChatResponse(session_id=session_id, reply="Invalid format. Use YYYY‑MM‑DD HH:MM.")
        if not is_valid_slot(slot_dt):
            return ChatResponse(
                session_id=session_id,
                reply="Please choose a time between 09:00–21:00 in 10‑minute increments."
            )
        # Check if requested slot is free
        exists = appointments_collection.find_one({
            "doctor_id": state["doctor"]["id"],
            "appointment_time": slot_dt
        })
        if exists:
            # Suggest next available slot
            next_slot = find_next_slot(state["doctor"]["id"], slot_dt + timedelta(minutes=10))
            state["next_slot"] = next_slot
            return ChatResponse(
                session_id=session_id,
                reply=(
                    f"That slot is taken. The next available is {next_slot}. "
                    "Would you like to book this one? (yes/no)"
                )
            )
        # Slot is free → ask for confirmation before booking
        state["proposed_slot"] = slot_dt
        state["stage"] = "confirm_slot"
        return ChatResponse(
            session_id=session_id,
            reply=(
                f"{slot_dt} is available. Confirm booking? (yes/no)"
            )
        )

    # CONFIRM_SLOT stage
    if state["stage"] == "confirm_slot":
        if user_msg.lower().startswith("y"):
            # finalize slot to book
            to_book = state.get("proposed_slot") or state.get("next_slot")
            appointments_collection.insert_one({
                "doctor_id": state["doctor"]["id"],
                "appointment_time": to_book,
                "patient_details": state["patient"]
            })
            name = state["doctor"]["name"]
            sessions.pop(session_id, None)
            return ChatResponse(
                session_id=session_id,
                reply=f"Your appointment is confirmed with Dr. {name} on {to_book}."
            )
        else:
            # user rejected proposed slot → back to await_slot
            state["stage"] = "await_slot"
            return ChatResponse(
                session_id=session_id,
                reply="Okay. Please provide another preferred slot (YYYY‑MM‑DD HH:MM)."
            )

    # Fallback
    sessions.pop(session_id, None)
    return ChatResponse(
        session_id=session_id,
        reply="An error occurred. Let's start over."
    )

# 9. Uvicorn entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
