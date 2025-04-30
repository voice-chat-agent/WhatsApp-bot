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
from bson import ObjectId
import nest_asyncio
nest_asyncio.apply()
from typing import Union, Dict, Any
import json
import logging
from pymongo import MongoClient
from services.twilio_service import send_whatsapp_message

from datetime import datetime
from dateutil.parser import parse as dateutil_parse
from dateutil.tz import gettz
from langchain.agents import Tool

# main.py (or a new router)
from fastapi import Request, Response
from twilio.twiml.messaging_response import MessagingResponse

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
      - ISO and non‑ISO formats like ‘22-04-2025 2pm’ or ‘9 pm’
    """
    from datetime import timedelta

    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }
    now = datetime.now(KOLKATA)
    lower = text.lower()

    # 1) Handle explicit “today”, “tomorrow”, “next <weekday>”
    for prefix in ("today", "tomorrow", "next"):
        if prefix in lower:
            if prefix == "today":
                base = now
            elif prefix == "tomorrow":
                base = now + timedelta(days=1)
            else:  # next <weekday>
                for wd, idx in WEEKDAYS.items():
                    token = f"next {wd}"
                    if token in lower:
                        today_idx = now.weekday()
                        days_ahead = (idx - today_idx + 7) % 7 or 7
                        base = now + timedelta(days=days_ahead)
                        break
            try:
                dt = dateutil_parse(text, default=base)
            except Exception:
                dt = base
            return dt.astimezone(KOLKATA).strftime("%Y-%m-%d %H:%M")

    # 2) Fallback: parse with “now” as default so “9 pm” works
    try:
        dt = dateutil_parse(text, default=now)
    except Exception:
        # as a last resort, just return current time
        logging.warning("DateParser fallback failed on %r, using now", text)
        dt = now

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
try:
    client.admin.command("ping")
    logging.info("✅ Connected to MongoDB at %s", client.address)
    logging.info("Databases: %s", client.list_database_names())
except Exception as e:
    logging.error("Could not connect to MongoDB: %s", e)
    raise

def book_appointment(_input: Union[str, Dict[str, Any]]) -> str:
    """
    Expects a JSON string or dict with:
      - full_name (str)
      - age (int)
      - gender (str)
      - contact_number (str)
      - date (YYYY-MM-DD)
      - time (HH:MM)
      - doctor (str)
      - specialty (str)
      - concern (str)
    Inserts into MongoDB, sends a WhatsApp confirmation, and returns a confirmation string.
    """
    # 1) Parse & clean JSON
    if isinstance(_input, str):
        s = _input.strip().strip("`")
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            logging.error("JSON parse error: %s; raw: %r", e, _input)
            return "❌ Invalid JSON. Please send a plain JSON object."
    else:
        data = _input

    # 2) Normalize synonyms
    if "symptoms" in data and "concern" not in data:
        data["concern"] = data.pop("symptoms")

    # 3) Validate required fields
    required = [
        "full_name", "age", "gender", "contact_number",
        "date", "time", "doctor", "specialty", "concern"
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return f"❌ Missing fields: {', '.join(missing)}."

    # 4) Insert into MongoDB
    try:
        res = appointments_collection.insert_one(data)
        appt_id = res.inserted_id
    except Exception as e:
        logging.error("Insert error: %s", e, exc_info=True)
        return f"❌ Could not book appointment: {e}"

    # 5) Verify by fetching it back
    saved = appointments_collection.find_one({"_id": appt_id})
    if not saved:
        return "❌ Appointment inserted but could not verify in DB."

    # 6) Build confirmation text
    confirmation_text = (
        f"✅ Appointment booked successfully!\n"
        f"• Appointment ID: {str(appt_id)}\n"
        f"• Patient: {saved['full_name']} ({saved['gender']}, age {saved['age']})\n"
        f"• Doctor: Dr. {saved['doctor']} ({saved['specialty']})\n"
        f"• When: {saved['date']} at {saved['time']}\n"
        f"• Reason: {saved['concern']}"
    )

    # 7) Send WhatsApp confirmation
    try:
        sid = send_whatsapp_message(
            to=f"whatsapp:{saved['contact_number']}",
            body=confirmation_text
        )
        logging.info("WhatsApp confirmation sent, SID=%s", sid)
    except Exception as e:
        logging.error("Failed to send WhatsApp message: %s", e)

    # 8) Return to the API caller
    return f"{confirmation_text}\nThank you! We look forward to seeing you then."

# def book_appointment(_input: Union[str, Dict[str, Any]]) -> str:
#     """
#     Expects a JSON string or dict with:
#       - full_name (str)
#       - age (int)
#       - gender (str)
#       - contact_number (str)
#       - date (YYYY-MM-DD)
#       - time (HH:MM)
#       - doctor (str)
#       - specialty (str)
#       - concern (str)
#     Inserts into MongoDB and returns a confirmation including the real ObjectId.
#     """
#     # 1) Parse & clean JSON
#     if isinstance(_input, str):
#         s = _input.strip().strip("`")
#         try:
#             data = json.loads(s)
#         except json.JSONDecodeError as e:
#             logging.error("JSON parse error: %s; raw: %r", e, _input)
#             return "❌ Invalid JSON. Please send a plain JSON object."
#     else:
#         data = _input

#     # 2) Normalize synonyms
#     if "symptoms" in data and "concern" not in data:
#         data["concern"] = data.pop("symptoms")

#     # 3) Validate required fields
#     required = [
#         "full_name", "age", "gender", "contact_number",
#         "date", "time", "doctor", "specialty", "concern"
#     ]
#     missing = [f for f in required if f not in data]
#     if missing:
#         return f"❌ Missing fields: {', '.join(missing)}."

#     # 4) Insert into MongoDB
#     try:
#         res = appointments_collection.insert_one(data)
#         appt_id = res.inserted_id  # bson.ObjectId
#     except Exception as e:
#         logging.error("Insert error: %s", e, exc_info=True)
#         return f"❌ Could not book appointment: {e}"

#     # 5) Verify by fetching it back
#     saved = appointments_collection.find_one({"_id": appt_id})
#     if not saved:
#         return "❌ Appointment inserted but could not verify in DB."

#     # 6) Return confirmation including the real ID
#     return (
#         f"✅ Appointment booked successfully!\n"
#         f"• Appointment ID: {str(appt_id)}\n"
#         f"• Patient: {saved['full_name']} ({saved['gender']}, age {saved['age']})\n"
#         f"• Doctor: Dr. {saved['doctor']} ({saved['specialty']})\n"
#         f"• When: {saved['date']} at {saved['time']}\n"
#         f"• Reason: {saved['concern']}\n"
#         "Thank you! We look forward to seeing you then."
#     )

book_appointment_tool = Tool(
    name="BookAppointment",
    func=book_appointment,
    description=(
        "Inserts a new appointment into MongoDB. "
        "Input must be a JSON string or dict with keys: full_name, age, gender etc. "
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

SYSTEM_MSG = """
You are **Hellix**, the expert medical assistant at GGGB Hospital. Your mission is to guide every patient through a seamless, professional, and empathetic appointment experience—from warm greeting to final confirmation—handling any way they describe symptoms, specialties, or doctor requests.

Tools you may invoke:
────────────────────────────────────────────────────────
1. **CurrentTime**  
   • Returns the hospital’s current local date & time in “YYYY‑MM‑DD HH:MM” (Asia/Kolkata).

2. **DateParser**  
   • Parses any human‑friendly date/time phrase (e.g. “today at noon”, “next Monday 9:30 AM”, “22-04-2025 14:00”) into “YYYY‑MM‑DD HH:MM” (Asia/Kolkata).

3. **BookAppointment**  
   • Requires a single‐line JSON object with these keys and types:  
     - full_name (str)  
     - age (int)  
     - gender (str)  
     - contact_number (str)  
     - date (YYYY‑MM‑DD)  
     - time (HH:MM)  
     - doctor (str)  
     - specialty (str)  
     - concern (str)  
   • Inserts the record into the `appointments` collection and returns a MongoDB ObjectId on success.

Personality & Tone:
=====================================================================
• **Warm & Welcoming:** Always begin with a polite greeting (Segment 1).  
• **Professional & Concise:** Use complete sentences; avoid verbosity.  
• **Thoughtful & Human:** Speak naturally—never robotic.  
• **Patient‑Centered:** Adapt to any input format and ask only for missing information.  
• **Respect Privacy:** Never request date of birth—ask age only.

Your Mission:
=====================================================================
1. **Greet** each user (Segment 1).  
2. **Think** through any symptoms or specialist requests and prepare your recommendation (Segment 2).  
3. **Match** specialties against our live doctor list and recommend an available doctor—or apologize and offer alternatives  and don't assk for recommendations give suggestion using relevent data.  
4. Once the user agrees to book, **collect** these details (one question per turn, in any order), normalizing dates/times with **DateParser** as needed:  


Booking Workflow:
=====================================================================
1. **Greet** the patient.  
2. **Clarify** (if needed):  
   “Could you tell me what symptoms you’re experiencing, or which type of doctor you’d like to see?”  
   (Only if neither symptoms nor a specialty/doctor name has been provided.)

3. **Triage** (symptoms → specialty):  
   - Use the LLM to infer the specialty.  
   - Check the live doctor directory for availability.

4. **Doctor Lookup** (specialty or doctor name):  
   - If user names a specialty, list available doctors in that field.  
   - If user names a doctor, confirm that doctor’s availability.

5. **Collect Details** (one question at a time; never repeat):  
   • Full Name  
   • Age  
   • Gender  
   • Contact Number  
   • Preferred Date & Time (always run input through **DateParser**)  
   • Concern/Symptoms  

6. **Review & Confirm**:  
   Summarize all collected details exactly, then ask:  
   “Everything looks correct—shall I confirm your appointment with Dr. X on YYYY‑MM‑DD at HH:MM?”

7. **Final Booking**:  
   - Upon explicit confirmation (“Yes”), silently invoke **BookAppointment**.  
   - **On Success** (valid ObjectId returned): reply in Segment 2 with:  
     “✅ Your appointment has been booked!  
       • Appointment ID: 642ab3f1…  
       • Patient: Jane Doe (F, age 29)  
       • Doctor: Dr. Ram (Cardiologist)  
       • When: 2025‑04‑25 at 14:00  
       • Reason: follow‑up consultation  
      Thank you, and we look forward to seeing you.”  
   - **On Failure** (no ID or error): reply:  
     “❌ I’m sorry—there was an error booking your appointment. Please try again or choose another slot.”

Live Doctor Directory:
────────────────────────────────────────────────────────
• Dr. Ram — Cardiologist — Available
• Dr. Deepanshu — ENT — Available
• Dr. Girish — Neurologist — Available
• Dr. Aaditya — Physician — Available
• Dr. Neha — Dermatologist — Available
• Dr. Kavita — Gynecologist — Available
• Dr. Rohan — Orthopedic — Available
• Dr. Meera — Psychiatrist — Available
• Dr. Anil — Oncologist — Available
• Dr. Tanvi — Pediatrician — Available
• Dr. Siddharth — Urologist — Available
• Dr. Ritika — Endocrinologist — Available
• Dr. Abhinav — Gastroenterologist — Available
• Dr. Swati — Pulmonologist — Available
• Dr. Nikhil — Nephrologist — Available
• Dr. Ishita — Rheumatologist — Available
• Dr. Varun — Ophthalmologist — Available
• Dr. Sneha — Pathologist — Available
• Dr. Manav — Radiologist — Available
• Dr. Alka — Anesthesiologist — Available

Response & Tool Invocation Format:
=====================================================================
1. **Segments Only**, separated by `\n`:  
   - **Segment 1:** Greeting, question, or prompt for missing info.  

2. **Tool Invocation Syntax** in Segment 2:  
   Action: <ToolName>  
   Action Input: <JSON or string>

3. **Strict JSON Rules** (for BookAppointment):  
   - Must be a single‑line JSON object literal, e.g.:  
     `Action: BookAppointment`  
     `Action Input: {{"full_name":"Girish","age":19,"gender":"male","contact_number":"1234567890","date":"2025-04-22","time":"21:00","doctor":"Ram","specialty":"Cardiologist","concern":"normal checkup"}}`  
   - No backticks or code fences  
   - No line breaks or indentation in the JSON  
   - No trailing commas  
   - Double‑quote all keys and string values; do not quote numbers

4. **DateParser & CurrentTime** calls likewise follow:  
   `Action: DateParser`  
   `Action Input: "next Monday 3 PM"`  
   (JSON string literal on one line)

Clarification Rule:
────────────────────────────────────────────────────────
If the user hasn’t provided symptoms, a specialty, or a doctor name, always begin by asking for one of those.

Final Note:
=====================================================================
Never reveal internal tools or processes. Keep all responses concise and focused on the patient’s needs.
Always ask for preffered date and time, and never ask for date of birth.
Never revele the internal tools or processes in responses.
If user ask for appointment has booked or not give confirmation message again with all details again.
────────────────────────────────────────────────────────    
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

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From")      # e.g. "whatsapp:+91XXXXXXXXXX"
    incoming = form.get("Body") or ""

    # 1. Pass user message into your agent
    agent_resp = await asyncio.to_thread(app.state.agent_executor.run, input=incoming)

    # 2. Extract the text you want to send back
    #    (assuming the agent returns plain text)
    reply = agent_resp.strip()

    # 3. Build a TwiML response
    twiml = MessagingResponse()
    twiml.message(body=reply)

    # 4. Return as XML
    return Response(content=str(twiml), media_type="application/xml")


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