# main.py
import warnings
warnings.filterwarnings('ignore')

import os
import uuid
import json
import logging
import asyncio

from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.tz import gettz

from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, PrivateAttr

from pymongo import MongoClient

import nest_asyncio
nest_asyncio.apply()

from datetime import datetime
from dateutil.parser import parse as dateutil_parse
from dateutil.tz import gettz
from langchain.agents import Tool


import json
from typing import Union, Dict, Any
from langchain.agents import Tool
#client = MongoClient("mongodb://localhost:27017")
client = MongoClient("mongodb+srv://Girish:Girish%402312@cluster0.l9vze.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
#from db.mongodb import db
db = client["doctor_db"]
doctors_collection      = db["doctors"]
appointments_collection = db["appointments"]


# ───────────────────────────────────────────────

# Timezone for all hospital timestamps
KOLKATA = gettz("Asia/Kolkata")

# 1) CurrentTime tool
def current_time(_input: str) -> str:
    now = datetime.now(KOLKATA)
    return now.strftime("%Y-%m-%d %H:%M")

current_time_tool = Tool(
    name="CurrentTime",
    func=current_time,
    description=(
        "Returns the current date & time in YYYY-MM-DD HH:MM "
        "format, anchored to Asia/Kolkata."
    )
)


def date_parser(text: str) -> str:
    """
    Takes any free‑form date/time phrase and returns a precise timestamp
    in “YYYY-MM-DD HH:MM” (Asia/Kolkata). Handles:
      - ‘today’, ‘tomorrow’, ‘next Monday’
      - ISO and non‑ISO formats like ‘22-04-2025 2pm’
    """
    # First, try to resolve relative keywords via a quick weekday handler
    from datetime import timedelta
    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }
    now = datetime.now(KOLKATA)
    lower = text.lower()
    for prefix in ("today", "tomorrow", "next"):
        if prefix in lower:
            # handle “today” and “tomorrow”
            if prefix == "today":
                base = now
            elif prefix == "tomorrow":
                base = now + timedelta(days=1)
            else:
                # next <weekday>
                for wd, idx in WEEKDAYS.items():
                    token = f"next {wd}"
                    if token in lower:
                        today_idx = now.weekday()
                        days_ahead = (idx - today_idx + 7) % 7 or 7
                        base = now + timedelta(days=days_ahead)
                        break
            # try extract a time with dateutil if present
            try:
                dt = dateutil_parse(text, default=base)
            except:
                dt = base
            return dt.astimezone(KOLKATA).strftime("%Y-%m-%d %H:%M")

    # Fallback: let dateutil parse absolute dates/times
    dt = dateutil_parse(text)
    dt = dt.astimezone(KOLKATA)
    return dt.strftime("%Y-%m-%d %H:%M")

date_parser_tool = Tool(
    name="DateParser",
    func=date_parser,
    description=(
        "Parses any human date/time phrase and returns a precise "
        "timestamp in YYYY-MM-DD HH:MM (Asia/Kolkata)."
    )
)



def ensure_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    s = x.strip().strip('“”"')
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"doctor": s}

def parse_appointment_time(time_str: str) -> str:
    dt = parse(time_str).astimezone(KOLKATA)
    return dt.strftime("%Y-%m-%d %H:%M")

def find_next_available_slot(doctor: str, requested: str) -> str:
    requested_dt = datetime.strptime(requested, "%Y-%m-%d %H:%M")
    for minutes in range(10, 24*60, 10):
        candidate = requested_dt + timedelta(minutes=minutes)
        slot_str = candidate.strftime("%Y-%m-%d %H:%M")
        if not appointments_collection.find_one({
            "doctor": doctor,
            "appointment_time": slot_str
        }):
            return slot_str
    raise ValueError("No available slots in next 24h")




# ───────────────────────────────────────────────────
# MongoDB Setup (blocking driver, wrapped later)
# ───────────────────────────────────────────────────
# client = MongoClient("mongodb://localhost:27017")
# db = client["doctor_db"]
# doctors_collection      = db["doctors"]
# appointments_collection = db["appointments"]

# ───────────────────────────────────────────────────
# Google GenAI LLM Wrapper
# ───────────────────────────────────────────────────
from langchain.llms.base import LLM
from google import genai

class GoogleGenAI(LLM):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    _api_key: str     = PrivateAttr()
    _client: genai.Client = PrivateAttr()

    def __init__(self, model_name: str, temperature: float, api_key: str):
        super().__init__()
        self.model_name  = model_name
        self.temperature = temperature
        self._api_key    = api_key
        self._client     = genai.Client(api_key=self._api_key)
        logging.info("GoogleGenAI initialized with %s", self.model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return resp.text

    @property
    def _llm_type(self) -> str:
        return "google_genai"

# ───────────────────────────────────────────────────
# LangChain Agent Setup
# ───────────────────────────────────────────────────
from langchain.agents import Tool, ConversationalAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Instantiate LLM
chat = GoogleGenAI(
    model_name=os.getenv("GENAI_MODEL_NAME", "gemini-2.0-flash"),
    temperature=float(os.getenv("GENAI_TEMPERATURE", "0.7")),
    api_key=os.getenv("GOOGLE_GENAI_API_KEY", "")
)
from langchain_ollama.llms import OllamaLLM
# chat = OllamaLLM(
#     model="gemma3:4b",
#     temperature=0.7
# )



import json
import logging
from typing import Union, Dict, Any
from langchain.agents import Tool

appointments_collection = db["appointments"]


import json
import logging
from typing import Union, Dict, Any
from langchain.agents import Tool
from pymongo import MongoClient
from typing import Union, Dict, Any
# — your actual URI here —

import json
import logging
from typing import Union, Dict, Any
from langchain.agents import Tool
from pymongo import MongoClient

client = MongoClient("mongodb+srv://Girish:Girish%402312@cluster0.l9vze.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["doctor_db"]
appointments_collection = db["appointments"]


def book_appointment(_input: Union[str, Dict[str, Any]]) -> str:
    # 1) Parse & clean the incoming string
    if isinstance(_input, str):
        s = _input.strip()
        # strip any leading/trailing backticks (``` or `)
        s = s.strip("`").strip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            logging.error("BookAppointment JSON parse error: %s\nRaw: %r", e, _input)
            return "❌ Invalid JSON. Please send a plain JSON object (no code fences)."
    else:
        data = _input

    # 2) Normalize 'symptoms' → 'concern'
    if "symptoms" in data and "concern" not in data:
        data["concern"] = data.pop("symptoms")

    # 3) Debug logging
    logging.info("BookAppointment payload: %s", data)
    print("📥 [DEBUG] book_appointment received:", data)

    # 4) Validate required fields
    required = [
        "full_name", "age", "gender", "contact_number",
        "date", "time", "doctor", "specialty", "concern"
    ]
    missing = [f for f in required if f not in data]
    if missing:
        msg = f"❌ Missing required fields: {', '.join(missing)}."
        logging.warning(msg)
        return msg

    # 5) Insert into MongoDB
    try:
        res = appointments_collection.insert_one(data)
        appt_id = str(res.inserted_id)
        logging.info("Inserted appointment _id=%s", appt_id)
        print("✅ [DEBUG] Inserted appointment _id=", appt_id)
    except Exception as e:
        logging.error("Error inserting appointment: %s", e, exc_info=True)
        return f"❌ Failed to book appointment: {e}"

    # 6) Return confirmation with Appointment ID
    return (
        f"✅ Appointment booked successfully!\n"
        f"• Appointment ID: {appt_id}\n"
        f"• Patient: {data['full_name']} ({data['gender']}, age {data['age']})\n"
        f"• Doctor: Dr. {data['doctor']} ({data['specialty']})\n"
        f"• When: {data['date']} at {data['time']}\n"
        f"• Reason: {data['concern']}\n"
        "Thank you! We look forward to seeing you then."
    )

book_appointment_tool = Tool(
    name="BookAppointment",
    func=book_appointment,
    description=(
        "Inserts a new appointment into MongoDB. "
        "Input must be a JSON string or dict with keys: full_name, age, gender, "
        "contact_number, date (YYYY-MM-DD), time (HH:MM), doctor, specialty, concern."
    )
)
# FAQ tool
def faq_llm(query: str) -> str:
    return chat(query)

faq_tool = Tool(
    name="IMA Hospital FAQ Bot",
    func=faq_llm,
    description="Answer general hospital FAQs via Gemini."
)

TOOLS = [
    faq_tool,
    current_time_tool,
    date_parser_tool,
    book_appointment_tool,
    # …any other tools…
]
# System prompt: includes delimiter instruction "|||"
SYSTEM_MSG = """
You are **Hellix**, the expert medical assistant at GGGB Hospital. You have three special tools:

1. **CurrentTime**  
   - Returns the hospital’s current local date & time in “YYYY‑MM‑DD HH:MM” (Asia/Kolkata).  
2. **DateParser**  
   - Parses any free‑form date/time phrase (e.g. “today at noon”, “next Monday 9:30 AM”, “22-04-2025 14:00”) and returns a precise timestamp in “YYYY‑MM‑DD HH:MM” (Asia/Kolkata), using **CurrentTime** as its reference.  
3. **BookAppointment**  
   - Accepts a JSON object with `full_name`, `age`, `gender`, `contact_number`, `date`, `time`, `doctor`, `specialty`, and `concern`, and inserts it into the `appointments` collection.

Your Personality & Tone:
=====================================================================
• **Warm & Welcoming:** Always start with a polite greeting (Segment 1).  
• **Thoughtful & Human:** Speak naturally and reassuringly—never robotic.  
• **Patient‑Centered:** Guide users step by step, adapting to any input format.

Your Mission:
=====================================================================
1. **Greet** each user (Segment 1).  
2. **Think** about any symptoms or specialist requests and prepare your recommendation (Segment 2).  
3. **Match** specialties against our live doctor list and recommend an available doctor—or apologize and offer alternatives.  
4. Once the user agrees to book, **collect** these details—one question per turn, in any order—and normalize with **DateParser** as needed:  
   a. Full Name  
   b. Age  
   c. Gender  
   d. Contact Number  
   e. Preferred Date (YYYY‑MM‑DD)  
   f. Preferred Time (HH:MM)  
   g. Concern/Symptoms  
   h. Any special instructions  
5. **Summarize** all details and ask:  
   “Everything looks correct. Should I confirm your appointment with Dr. X on [Date] at [Time]?”

Live Doctor Directory:
    • Dr. Ram        — Cardiologist   — Available  
    • Dr. Deepanshu  — ENT            — Available  
    • Dr. Girish     — Neurologist    — Available  
    • Dr. Aaditya    — Physician      — Available  

Response Format:
=====================================================================
Send **exactly two segments**, separated by `|||`:
- **Segment 1:** when it need to give two differnte replies which are not connnected  
- **Segment 2:** Your recommendation, next step, or tool invocation.

Booking Flow & Tools:
=====================================================================
- For any relative or non‑ISO date/time (“today,” “tomorrow,” “next Friday,” “2 PM”), **always** call `DateParser(user_text)`.  
- If you need the current timestamp, call `CurrentTime()`.  
- Prompt only for missing fields—do **not** repeat information you already have.

Finalization:
=====================================================================
When the user says **“Yes”** to “Everything looks correct. Should I confirm your appointment with Dr. X on [Date] at [Time]?”, you **must** perform exactly and use the booking tool to insert the details":
After booking also give booking ID and all the details of the appointment.
Never tell about what tools you are using or how you are using them in your response.
Don't give too much long response.
Don't ask DOB just age

"""

HUMAN_MSG = """
{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ConversationalAgent.create_prompt(
    TOOLS,
    prefix=SYSTEM_MSG,
    suffix=HUMAN_MSG,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=chat, prompt=prompt)

def get_agent_executor() -> AgentExecutor:
    agent  = ConversationalAgent(
        llm_chain=llm_chain,
        tools=TOOLS,
        verbose=True,
        return_intermediate_steps=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )

# ───────────────────────────────────────────────────
# FastAPI Application
# ───────────────────────────────────────────────────
app = FastAPI()

class ChatRequest(BaseModel):
    input: str
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    responses: List[str]       # split segments here
    chat_history: List[Dict[str, str]]

session_histories: Dict[str, List[Dict[str, str]]] = {}

@app.on_event("startup")
async def startup_event():
    app.state.agent_executor = get_agent_executor()
    logging.info("AgentExecutor initialized")

@app.get("/new_session")
async def new_session():
    sid = str(uuid.uuid4())
    session_histories[sid] = []
    return {"session_id": sid}

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(session_id: str, req: ChatRequest):
    # Load or init history
    if session_id not in session_histories:
        session_histories[session_id] = req.chat_history
    full_hist = session_histories[session_id] + req.chat_history

    # Convert to LangChain messages
    msg_objs = []
    for m in full_hist:
        if m["role"] == "user":
            msg_objs.append(HumanMessage(content=m["content"]))
        else:
            msg_objs.append(AIMessage(content=m["content"]))

    # Inject into memory
    agent_exec: AgentExecutor = app.state.agent_executor
    agent_exec.memory.chat_memory.messages = msg_objs

    # Run agent (non-blocking)
    raw = await asyncio.to_thread(agent_exec.run, input=req.input)

    # Split on delimiter
    parts = [p.strip() for p in raw.split("|||") if p.strip()]
    if not parts:
        parts = [raw.strip()]

    # Update history
    updated = []
    for msg in agent_exec.memory.chat_memory.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        updated.append({"role": role, "content": msg.content})
    session_histories[session_id] = updated

    return ChatResponse(responses=parts, chat_history=updated)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="app.log",
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
