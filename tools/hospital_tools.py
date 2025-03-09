# tools/hospital_tools.py
import logging
from langchain.agents import Tool
from services.doctor_lookup import lookup_doctor_or_appointment
from services.appointment import book_appointment
from utils.helpers import ensure_dict
from services.faq_tool import create_faq_tool  # Changed import here

# Create the FAQ tool using the new module
faq_tool = create_faq_tool()

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
    from services.appointment import appointments_collection  # local import to avoid circular dependency
    query = ensure_dict(query)
    appointments = appointments_collection.find(query)
    result = []
    for appointment in appointments:
        appointment["_id"] = str(appointment["_id"])
        result.append(appointment)
    return result

appointment_search_tool = Tool(
    name="IMA Hospital Appointment Search",
    func=search_appointment,
    description="Use this tool to search for appointments. The query should include keys like 'doctor', 'appointment_time', or the patient's name."
)

doctor_lookup_tool = Tool(
    name="IMA Hospital Doctor Lookup",
    func=lookup_doctor_or_appointment,
    description="Use this tool to retrieve available doctor details by name or specialist from our MongoDB Atlas database."
)

# Assemble the list of tools
TOOLS = [faq_tool, doctor_lookup_tool, appointment_booking_tool, appointment_search_tool]
