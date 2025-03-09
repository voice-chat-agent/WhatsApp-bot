# services/doctor_lookup.py
import re
import logging
from db.mongodb import db

doctors_collection = db["doctors"]

def lookup_doctor_or_appointment(query: str) -> str:
    logging.info("lookup_doctor_or_appointment called with query: %s", query)
    match = re.search(r"Dr\.?\s+([A-Za-z]+)", query, re.IGNORECASE)
    if match:
        doctor_name = match.group(1)
        doctor = doctors_collection.find_one({
            "name": {"$regex": doctor_name, "$options": "i"},
            "availability": True
        })
        if doctor:
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
            return "No available doctor details found for that name."
    else:
        specializations = ["cardiologist", "dermatologist", "neurologist", "pediatrician", "orthopedic", "oncologist"]
        for spec in specializations:
            if spec in query.lower():
                doctor = doctors_collection.find_one({
                    "specialty": {"$regex": spec, "$options": "i"},
                    "availability": True
                })
                if doctor:
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
                    return f"No available doctor found for {spec}."
        return "No specific doctor information detected in your query."
