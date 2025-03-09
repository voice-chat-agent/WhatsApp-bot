# main.py
import warnings
warnings.filterwarnings('ignore')

import os
import re
import nest_asyncio
from datetime import datetime, timedelta
import uuid
import json
import logging

from dateutil.parser import parse  # To handle loosely formatted dates

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

from typing import List
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# --------------------------
# LangChain & Related Imports
# --------------------------
from langchain.agents import Tool, ConversationalAgent, AgentExecutor
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory  # Using full conversation history
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage  # For converting dicts to message objects
from langchain.llms.ollama import Ollama
from langchain_ollama import OllamaLLM

# --------------------------
# Pinecone Setup for RetrievalQA
# --------------------------
from langchain_pinecone import PineconeVectorStore

# --------------------------
# Environment Variables
# (Ensure these are defined in your .env file or replace with your keys)
# --------------------------
os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_5bC2FA_SwRF3C5ghLX1iGhubvU9sdc7TVqkWnvgczN4tMb43nFNuXdT7iTLsFivPzsJsEP"
logging.info("API keys loaded.")
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name='healthcare-kb',  # Update with your actual index name
    embedding=embeddings
)

retriever = vectorstore.as_retriever()
chat = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=False)
logging.info("Model loaded.")
#ollama_llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.7)
# Build a RetrievalQA chain for general hospital FAQs
qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

# qa = RetrievalQA.from_chain_type(
#     llm=ollama_llm,
#     chain_type="stuff",
#     retriever=retriever
# )

logging.info("Retriever initialized.")
faq_tool = Tool(
    name="IMA Hospital FAQ Bot",
    func=qa.run,
    description="Use this tool for general hospital information such as services, directions, or FAQs."
)
logging.info("FAQ tool created.")

# --------------------------
# MongoDB Setup Atlas
# --------------------------
from db.mongodb import db  
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]  # Ensure this collection exists
logging.info("MongoDB connected and collections loaded.")
print("___________________________________________________________")
print("___________________________________________________________")
print("___________________________________________________________")
if "doctors" in db.list_collection_names():
    print("The 'doctors' collection exists.")
else:
    print("The 'doctors' collection does not exist.")

if "appointments" in db.list_collection_names():
    print("The 'appointments' collection exists.")
else:
    print("The 'appointments' collection does not exist.")
print("___________________________________________________________")
print("___________________________________________________________")
print("___________________________________________________________")

def lookup_doctor_or_appointment(query: str) -> str:
    logging.info("lookup_doctor_or_appointment called with query: %s", query)
    # First, try to match a doctor by name, filtering on availability.
    match = re.search(r"Dr\.?\s+([A-Za-z]+)", query, re.IGNORECASE)
    if match:
        doctor_name = match.group(1)
        logging.info("Doctor name extracted: %s", doctor_name)
        doctor = doctors_collection.find_one({
            "name": {"$regex": doctor_name, "$options": "i"},
            "availability": True
        })
        logging.info("Database query for doctor by name executed.")
        if doctor:
            logging.info("Doctor found: %s", doctor.get('name', 'N/A'))
            languages = doctor.get('languages', [])
            lang_str = ', '.join(languages) if languages else "N/A"
            return (
                f"Doctor Details:\n"
                f"Name: {doctor.get('name', 'N/A')}\n"
                f"Specialty: {doctor.get('specialty', 'N/A')}\n"
                f"Languages: {lang_str}\n"
                f"Availability: {doctor.get('availability', 'Not Available')}"
            )
        else:
            logging.info("No available doctor found by name: %s", doctor_name)
            return "No available doctor details found for that name."
    else:
        logging.info("No doctor name found in query, checking for specialization.")
        # Attempt to search by specialist keyword using the "specialty" field.
        specializations = ["cardiologist", "dermatologist", "neurologist", "pediatrician", "orthopedic", "oncologist"]
        for spec in specializations:
            if spec in query.lower():
                logging.info("Specialization '%s' found in query.", spec)
                doctor = doctors_collection.find_one({
                    "specialty": {"$regex": spec, "$options": "i"},
                    "availability": True
                })
                if doctor:
                    logging.info("Doctor with specialization %s found: %s", spec, doctor.get('name', 'N/A'))
                    languages = doctor.get('languages', [])
                    lang_str = ', '.join(languages) if languages else "N/A"
                    return (
                        f"Doctor Details:\n"
                        f"Name: {doctor.get('name', 'N/A')}\n"
                        f"Specialty: {doctor.get('specialty', 'N/A')}\n"
                        f"Languages: {lang_str}\n"
                        f"Availability: {doctor.get('availability', 'Not Available')}"
                    )
                else:
                    logging.info("No available doctor found for specialization: %s", spec)
                    return f"No available doctor found for {spec}."
        logging.info("No specific doctor information detected in query.")
        return "No specific doctor information detected in your query."

def find_next_available_slot(doctor: str, requested_dt: datetime) -> str:
    logging.info("find_next_available_slot called for doctor: %s at requested time: %s", doctor, requested_dt)
    closing_dt = requested_dt.replace(hour=21, minute=0)
    current_dt = requested_dt + timedelta(minutes=10)
    while current_dt < closing_dt:
        slot_str = current_dt.strftime("%Y-%m-%d %H:%M")
        logging.debug("Checking slot: %s", slot_str)
        if not appointments_collection.find_one({
            "doctor": doctor,
            "appointment": slot_str
        }):
            logging.info("Next available slot found: %s", slot_str)
            return slot_str
        current_dt += timedelta(minutes=10)
    logging.info("No available slot found on that day for doctor: %s", doctor)
    return None

def parse_appointment_time(time_str: str) -> str:
    logging.info("parse_appointment_time called with input: %s", time_str)
    try:
        # First, try the strict format.
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        logging.info("Time parsed using strict format: %s", dt)
    except Exception as e:
        logging.info("Strict parsing failed, attempting fuzzy parsing: %s", e)
        # If strict parsing fails, use dateutil's parser with a default date set to January 1, 2025.
        default_dt = datetime(2025, 1, 1)
        dt = parse(time_str, fuzzy=True, default=default_dt)
        logging.info("Time parsed using fuzzy parsing: %s", dt)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
    logging.info("Formatted appointment time: %s", formatted_time)
    return formatted_time

def ensure_dict(input_data):
    logging.info("ensure_dict called with input data type: %s", type(input_data))
    if isinstance(input_data, dict):
        return input_data
    elif isinstance(input_data, str):
        try:
            # First, try to parse it as JSON.
            result = json.loads(input_data)
            logging.info("Input string successfully parsed as JSON.")
            return result
        except Exception as e:
            logging.info("JSON parsing failed (%s), attempting naive conversion.", e)
            # Fallback: Split by commas and colons.
            result = {}
            for part in input_data.split(","):
                if ":" in part:
                    key, val = part.split(":", 1)
                    result[key.strip().lower()] = val.strip()
            logging.info("Naive conversion result: %s", result)
            return result
    else:
        logging.error("Unexpected type for appointment details: %s", type(input_data))
        raise ValueError("Unexpected type for appointment details. Expected dict or str.")

# def book_appointment(appointment_details: dict) -> str:
#     logging.info("book_appointment called with details: %s", appointment_details)
#     # Ensure appointment_details is a dict.
#     appointment_details = ensure_dict(appointment_details)
    
#     # Parse the appointment_time into the required format.
#     try:
#         formatted_time = parse_appointment_time(appointment_details["appointment_time"])
#         appointment_details["appointment_time"] = formatted_time
#         requested_dt = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M")
#         logging.info("Appointment time parsed and formatted: %s", formatted_time)
#     except Exception as e:
#         logging.error("Error parsing appointment time: %s", e)
#         return str(e)

#     # Check that the appointment time is within hospital working hours (9 AM to 9 PM)
#     opening_dt = requested_dt.replace(hour=9, minute=0)
#     closing_dt = requested_dt.replace(hour=21, minute=0)
#     if not (opening_dt <= requested_dt < closing_dt):
#         logging.info("Requested time %s is outside hospital hours.", formatted_time)
#         return "Appointment time must be within hospital working hours (9 AM to 9 PM)."

#     # Check that the requested doctor/specialist is available (only available doctors are considered).
#     doctor_query = appointment_details.get("doctor")
#     if doctor_query:
#         logging.info("Doctor query provided: %s", doctor_query)
#         available_doctor = doctors_collection.find_one({
#             "$and": [
#                 {"availability": True},
#                 {"$or": [
#                     {"name": {"$regex": doctor_query, "$options": "i"}},
#                     {"specialty": {"$regex": doctor_query, "$options": "i"}}
#                 ]}
#             ]
#         })
#         if not available_doctor:
#             logging.info("No available doctor found for query: %s", doctor_query)
#             return f"I'm sorry, but we currently do not have any available doctors for '{doctor_query}'. Please check our available specialists or try a different specialty."
#         else:
#             logging.info("Available doctor found: %s", available_doctor.get('name'))
#     else:
#         logging.info("No doctor query provided in appointment details.")

#     # Check if the requested slot is already booked
#     existing_appointment = appointments_collection.find_one({
#          "doctor": appointment_details.get("doctor"),
#          "appointment_time": appointment_details.get("appointment_time")
#     })
#     if existing_appointment:
#         logging.info("Requested slot %s for doctor %s is already booked.", appointment_details.get("appointment_time"), appointment_details.get("doctor"))
#         next_slot = find_next_available_slot(appointment_details.get("doctor"), requested_dt)
#         if next_slot:
#             logging.info("Suggesting next available slot: %s", next_slot)
#             return f"The requested slot is already booked. Would you like to book an appointment at {next_slot} instead?"
#         else:
#             logging.info("No alternative slot available for the day.")
#             return "The requested slot is already booked and no alternative slot is available on that day. Please choose another day."
    
#     # Book the appointment
#     result = appointments_collection.insert_one(appointment_details)
#     logging.info("Inserted appointment with ID: %s", result.inserted_id)
#     print("___________________________________________________________")
#     print("___________________________________________________________")
#     print("success")
#     print("___________________________________________________________")
#     print("___________________________________________________________")
#     return f"Appointment booked successfully with ID: {result.inserted_id}. Details: {appointment_details}"

# def book_appointment(appointment_details: dict) -> str:
#     logging.info("book_appointment called with details: %s", appointment_details)
#     # Ensure appointment_details is a dict.
#     appointment_details = ensure_dict(appointment_details)
    
#     # Parse the appointment_time into the required format.
#     try:
#         formatted_time = parse_appointment_time(appointment_details["appointment_time"])
#         appointment_details["appointment_time"] = formatted_time
#         requested_dt = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M")
#         logging.info("Appointment time parsed and formatted: %s", formatted_time)
#     except Exception as e:
#         logging.error("Error parsing appointment time: %s", e)
#         return str(e)

#     # Check that the appointment time is within hospital working hours (9 AM to 9 PM)
#     opening_dt = requested_dt.replace(hour=9, minute=0)
#     closing_dt = requested_dt.replace(hour=21, minute=0)
#     if not (opening_dt <= requested_dt < closing_dt):
#         logging.info("Requested time %s is outside hospital hours.", formatted_time)
#         return "Appointment time must be within hospital working hours (9 AM to 9 PM)."

#     # Verify that the requested doctor/specialist is available (only available doctors are considered).
#     doctor_query = appointment_details.get("doctor")
#     if doctor_query:
#         logging.info("Doctor query provided: %s", doctor_query)
#         available_doctor = doctors_collection.find_one({
#             "$and": [
#                 {"availability": True},
#                 {"$or": [
#                     {"name": {"$regex": doctor_query, "$options": "i"}},
#                     {"specialty": {"$regex": doctor_query, "$options": "i"}}
#                 ]}
#             ]
#         })
#         if not available_doctor:
#             logging.info("No available doctor found for query: %s", doctor_query)
#             return f"I'm sorry, but we currently do not have any available doctors for '{doctor_query}'. Please check our available specialists or try a different specialty."
#         else:
#             logging.info("Available doctor found: %s", available_doctor.get('name'))
#     else:
#         logging.info("No doctor query provided in appointment details.")

#     # Directly book the appointment without checking if the slot is already booked.
#     try:
#         result = appointments_collection.insert_one(appointment_details)
#         if result.inserted_id:
#             ack = f"Appointment booked successfully with ID: {result.inserted_id}. Details: {appointment_details}"
#             logging.info("Appointment booking acknowledged: %s", ack)
#             return ack
#         else:
#             logging.error("No acknowledgment from booking, please try again.")
#             return "Booking failed, please try again."
#     except Exception as e:
#         logging.error("Exception during appointment booking: %s", e)
#         return "Error booking appointment, please try again."

KEY_MAPPING = {
    "appointment date and time": "appointment_time",
    "appointment_date_time": "appointment_time",
    "appointment_time": "appointment_time",
    "doctor name": "doctor",
    "dr": "doctor",
    "doctor": "doctor",
    "symptom description": "symptom_description",
    "symptom_description": "symptom_description",
    "patient details": "patient",
    "patient info": "patient",
    "patient": "patient"
}

def normalize_details(details: dict) -> dict:
    # Convert keys to lower case and map to standard names
    return { KEY_MAPPING.get(key.lower(), key): value for key, value in details.items() }

def book_appointment(appointment_details: dict) -> str:
    logging.info("book_appointment called with details: %s", appointment_details)
    appointment_details = ensure_dict(appointment_details)
    # Normalize keys using a generalized mapping
    appointment_details = normalize_details(appointment_details)
    logging.info("Normalized details: %s", appointment_details)
    
    # Parse appointment_time
    try:
        formatted_time = parse_appointment_time(appointment_details["appointment_time"])
        appointment_details["appointment_time"] = formatted_time
        requested_dt = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M")
        logging.info("Parsed appointment time: %s", formatted_time)
    except Exception as e:
        logging.error("Error parsing appointment time: %s", e)
        return str(e)
    
    # Verify working hours (9 AM to 9 PM)
    opening_dt = requested_dt.replace(hour=9, minute=0)
    closing_dt = requested_dt.replace(hour=21, minute=0)
    if not (opening_dt <= requested_dt < closing_dt):
        logging.info("Time %s outside working hours", formatted_time)
        return "Appointment time must be within hospital working hours (9 AM to 9 PM)."
    
    # Book appointment (direct insert)
    try:
        result = appointments_collection.insert_one(appointment_details)
        if result.inserted_id:
            ack = f"Appointment booked successfully with ID: {result.inserted_id}. Details: {appointment_details}"
            logging.info("Booking acknowledged: %s", ack)
            return ack
        else:
            logging.error("No booking acknowledgment received.")
            return "Booking failed, please try again."
    except Exception as e:
        logging.error("Exception during booking: %s", e)
        return "Error booking appointment, please try again."


appointment_booking_tool = Tool(
    name="IMA Hospital Appointment Booking",
    func=book_appointment,
    description=(
        "Use this tool to book an appointment. Provide complete details in JSON format including doctor's name or specialty, "
        "patient's symptom description, appointment date and time (the tool accepts loosely formatted inputs and converts them to YYYY-MM-DD HH:MM, defaulting the year to 2025 if missing), and patient details "
        "(full name, age, gender, contact number, optionally email). Note: Hospital working hours are 9 AM to 9 PM, "
        "and appointments are available every 10 minutes."
    )
)

def search_appointment(query: dict) -> list:
    logging.info("search_appointment called with query: %s", query)
    query = ensure_dict(query)
    appointments = appointments_collection.find(query)
    result = []
    for appointment in appointments:
        appointment["_id"] = str(appointment["_id"])
        result.append(appointment)
    logging.info("search_appointment returning %d appointments", len(result))
    return result

appointment_search_tool = Tool(
    name="IMA Hospital Appointment Search",
    func=search_appointment,
    description="Use this tool to search for appointments. The query should include keys like 'doctor', 'appointment_time', or the patient's name."
)

# --------------------------
# Assemble All Tools
# --------------------------
tools = [
    faq_tool, 
    Tool(
        name="IMA Hospital Doctor Lookup",
        func=lookup_doctor_or_appointment,
        description="Use this tool to retrieve available doctor details by name or specialist from our MongoDB Atlas database."
    ),
    appointment_booking_tool,
    appointment_search_tool
]

# --------------------------
# Revised System Prompt with Detailed Instructions
# --------------------------
system_message = """
You are Helix, a customer-oriented, chain-of-thought AI assistant for IMA Hospital.
When a query is received, analyze it step by step:
- If the query is about general hospital information (services, directions, FAQs), use the "IMA Hospital FAQ Bot" tool.
- If the query is about checking doctor details or appointment availability, use the "IMA Hospital Doctor Lookup" tool.
- If the query includes a specialist request (e.g., "cardiologist", "dermatologist"), search the MongoDB database for an available doctor with that specialty and return the doctor's details if available. Do not rely solely on internal knowledge. If no such available doctor is found, gently inform the user.
- If the query comes with a specific appointment time slot (formatted as YYYY-MM-DD HH:MM or loosely formatted), check the appointments database to see if that slot is already booked.
    * If the slot is booked, inform the user and suggest the next available 10-minute slot.
    * Otherwise, proceed with the booking.
- If the query mentions concerning symptoms (e.g., chest pain) but does NOT explicitly state "I want to book an appointment", first ask clarifying questions before suggesting an appointment.
- Note: The hospital operates only from 9 AM to 9 PM. Appointments are available in 10-minute intervals.
- Note: Before booking the appointment, always ask for confirmation.
- When booking appointments, verify that all required details are provided:
    * Doctor's name or specialty.
    * Patient's symptom description.
    * Appointment date and time – the tool accepts loosely formatted inputs and converts them to the standard format (YYYY-MM-DD HH:MM), with the year defaulting to 2025 if missing.
    * Patient details: full name, age, gender, contact number (and optionally email).
- The agent can access real-time information to book future appointments.
- The agent can use tool to book an appointments.
- If necessary, use the "IMA Hospital Appointment Search" tool to verify existing appointments.
Combine the results from the tools and provide a concise, informative final answer.
Maintain a polite and professional tone.
"""

human_message = """
Begin!

{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=system_message,
    suffix=human_message,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
#gpt-deepseek
llm_chain = LLMChain(llm=chat, prompt=prompt)
#llm_chain = LLMChain(llm=ollama_llm, prompt=prompt)

def get_agent_executor() -> AgentExecutor:
    logging.info("Creating agent executor with full conversation memory.")
    agent = ConversationalAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    #gpt-deepseek
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )

    # agent_executor = AgentExecutor.from_agent_and_tools(
    # agent=agent,
    # tools=tools,
    # verbose=True,
    # memory=memory,
    # handle_parsing_errors=True
    # )

    logging.info("Agent executor created successfully.")
    return agent_executor

# --------------------------
# FastAPI Application Setup with Session Support
# --------------------------
app = FastAPI()

class ChatRequest(BaseModel):
    input: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[dict]

# Global dictionary to hold session histories.
session_histories = {}

@app.get("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())
    session_histories[session_id] = []  # Initialize an empty history for this session.
    logging.info("New session created with session_id: %s", session_id)
    return {"session_id": session_id}

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(session_id: str, chat_request: ChatRequest):
    logging.info("chat_endpoint called for session_id: %s", session_id)
    # Retrieve existing history or initialize a new one if not found.
    if session_id not in session_histories:
        session_histories[session_id] = chat_request.chat_history
        logging.info("Session history not found. Initializing new history for session_id: %s", session_id)

    # Merge any new history from the request with the stored history.
    full_history = session_histories[session_id] + chat_request.chat_history
    logging.info("Merged chat history length for session_id %s: %d", session_id, len(full_history))

    # Convert full_history from list of dicts to list of LangChain message objects.
    def convert_dict_to_message(item):
        role = item.get("role")
        content = item.get("content")
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)
    
    message_objects = [convert_dict_to_message(item) for item in full_history if isinstance(item, dict)]
    logging.info("Converted chat history to message objects for session_id: %s", session_id)

    agent_executor = get_agent_executor()
    # Load the full conversation history (as message objects) into memory.
    agent_executor.memory.chat_memory.messages = message_objects

    response = agent_executor.run(chat_request.input)
    logging.info("Agent executor completed processing for session_id: %s", session_id)

    # Convert conversation memory messages back to plain dictionaries.
    updated_history = []
    for msg in agent_executor.memory.chat_memory.messages:
        if isinstance(msg, dict):
            updated_history.append(msg)
        else:
            msg_type = msg.__class__.__name__
            if msg_type == "HumanMessage":
                role = "user"
            elif msg_type == "AIMessage":
                role = "assistant"
            else:
                role = "unknown"
            updated_history.append({"role": role, "content": msg.content})
    logging.info("Updated chat history length for session_id %s: %d", session_id, len(updated_history))
    
    # Update the session history.
    session_histories[session_id] = updated_history

    return ChatResponse(response=response, chat_history=updated_history)

if __name__ == "__main__":
    logging.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
