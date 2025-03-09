# services/appointment.py
import logging
from datetime import datetime, timedelta
from db.mongodb import db
from utils.helpers import ensure_dict, parse_appointment_time

appointments_collection = db["appointments"]

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
    return { KEY_MAPPING.get(key.lower(), key): value for key, value in details.items() }

def find_next_available_slot(doctor: str, requested_dt: datetime) -> str:
    closing_dt = requested_dt.replace(hour=21, minute=0)
    current_dt = requested_dt + timedelta(minutes=10)
    while current_dt < closing_dt:
        slot_str = current_dt.strftime("%Y-%m-%d %H:%M")
        if not appointments_collection.find_one({
            "doctor": doctor,
            "appointment": slot_str
        }):
            return slot_str
        current_dt += timedelta(minutes=10)
    return None

def book_appointment(appointment_details: dict) -> str:
    appointment_details = ensure_dict(appointment_details)
    appointment_details = normalize_details(appointment_details)
    
    try:
        formatted_time = parse_appointment_time(appointment_details["appointment_time"])
        appointment_details["appointment_time"] = formatted_time
        requested_dt = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M")
    except Exception as e:
        logging.error("Error parsing appointment time: %s", e)
        return str(e)
    
    opening_dt = requested_dt.replace(hour=9, minute=0)
    closing_dt = requested_dt.replace(hour=21, minute=0)
    if not (opening_dt <= requested_dt < closing_dt):
        return "Appointment time must be within hospital working hours (9 AM to 9 PM)."
    
    try:
        result = appointments_collection.insert_one(appointment_details)
        if result.inserted_id:
            return f"Appointment booked successfully with ID: {result.inserted_id}. Details: {appointment_details}"
        else:
            return "Booking failed, please try again."
    except Exception as e:
        logging.error("Exception during booking: %s", e)
        return "Error booking appointment, please try again."
