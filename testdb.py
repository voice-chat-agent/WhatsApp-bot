# from pymongo import MongoClient
# import re

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017")
# doctor_db = client["doctor_db"]
# doctors_collection = doctor_db["doctors"]

# def get_doctor_info(query: str) -> str:
#     query_clean = query.strip()

#     # Try matching by doctor name (exact, case-insensitive)
#     doctor_match = doctors_collection.find_one({
#         "name": re.compile(f"^{re.escape(query_clean)}$", re.IGNORECASE)
#     })

#     if doctor_match:
#         specialty = doctor_match.get("specialty", "Unknown")
#         availability = "Yes" if doctor_match.get("availability") else "No"
#         return f"Doctor: {doctor_match['name'].title()} | Specialty: {specialty.title()} | Available: {availability}"

#     # Else, try finding by specialty (regex match)
#     specialty_matches = doctors_collection.find({
#         "specialty": re.compile(f".*{re.escape(query_clean)}.*", re.IGNORECASE)
#     })

#     results = list(specialty_matches)
#     if results:
#         output = [f"Doctors with specialty matching '{query_clean.title()}':"]
#         for doc in results:
#             avail = "Yes" if doc.get("availability") else "No"
#             output.append(f"- {doc['name'].title()} (Available: {avail})")
#         return "\n".join(output)

#     return f"No results found for '{query_clean}'."


# # ---- Interactive input ----
# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter doctor name or specialty (or type 'exit' to quit): ").strip()
#         if user_input.lower() == "exit":
#             break
#         response = get_doctor_info(user_input)
#         print(response)
#         print("-" * 50)
from pymongo import MongoClient
from pprint import pprint

# 1. Connect to MongoDB and select the 'appointments' collection
client = MongoClient("mongodb://localhost:27017")
db = client["doctor_db"]
appointments = db["appointments"]

def debug_print_appointments():
    # 2. Check how many documents are in the collection
    total = appointments.count_documents({})
    print(f"Total documents in appointments collection: {total}\n")
    
    # 3. If there are any, print them; otherwise note that it’s empty
    if total == 0:
        print("⚠️  No appointments found. The collection is empty.")
    else:
        print("Listing all appointment documents:\n")
        for doc in appointments.find():
            pprint(doc)

if __name__ == "__main__":
    debug_print_appointments()
