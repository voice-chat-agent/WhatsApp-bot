from pymongo import MongoClient

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017")

# Use the database and collection
db = client["doctor_db"]
doctors = db["doctors"]

# Prepare doctor documents
doctor_data = [
    {
        "_id": "67afaa8f29e2f3050eb651fc",
        "tenant_id": "global",
        "doctor_id": 34,
        "name": "ram",
        "specialty": "cardiologist",
        "languages": [],
        "about": "sdf",
        "clinic_interests": "sd",
        "education": "df",
        "personal_interests": "sdf",
        "availability": True
    },
    {
        "_id": "67f7f4fcfa3e2bb2780bfcb9",
        "tenant_id": "global",
        "doctor_id": 2,
        "name": "Deepanshu",
        "specialty": "ENT",
        "languages": [],
        "about": "sdg",
        "clinic_interests": "ENT",
        "education": "df",
        "personal_interests": "sdf",
        "availability": True
    },
    {
        "_id": "67f7f60bfa3e2bb2780bfcbb",
        "tenant_id": "global",
        "doctor_id": 3,
        "name": "Girish",
        "specialty": "Neurologist",
        "languages": [],
        "about": "sdg",
        "clinic_interests": "Neurologist",
        "education": "df",
        "personal_interests": "sdf",
        "availability": True
    },
    {
        "_id": "67f7f619fa3e2bb2780bfcbd",
        "tenant_id": "global",
        "doctor_id": 4,
        "name": "aaditya",
        "specialty": "physician",
        "languages": [],
        "about": "sdg",
        "clinic_interests": "physician",
        "education": "df",
        "personal_interests": "sdf",
        "availability": True
    }
]

# Insert into collection (use insert_many if not already inserted)
try:
    doctors.insert_many(doctor_data, ordered=False)
    print("Doctor data inserted successfully.")
except Exception as e:
    print("Some records may already exist or error occurred:", e)


from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
doctor_db = client["doctor_db"]
appointments = doctor_db["appointments"]

# Sample appointment document
appointment_record = {
    "doctor": "Dr. Ram",
    "symptom_description": "Chest pain",
    "appointment_time": datetime(2025, 4, 22, 10, 0),
    "patient_details": {
        "name": "John Doe",
        "age": 45,
        "phone": "1234567890"
    }
}

# Insert into appointments collection
inserted = appointments.insert_one(appointment_record)
print(f"Inserted appointment with _id: {inserted.inserted_id}")
