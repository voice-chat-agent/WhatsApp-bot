# utils/helpers.py
import json
from dateutil.parser import parse
from datetime import datetime

def ensure_dict(input_data):
    if isinstance(input_data, dict):
        return input_data
    elif isinstance(input_data, str):
        try:
            return json.loads(input_data)
        except Exception:
            result = {}
            for part in input_data.split(","):
                if ":" in part:
                    key, val = part.split(":", 1)
                    result[key.strip().lower()] = val.strip()
            return result
    else:
        raise ValueError("Unexpected type. Expected dict or str.")

def parse_appointment_time(time_str: str) -> str:
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except Exception:
        # If strict parsing fails, use fuzzy parsing with default date of 2025-01-01.
        dt = parse(time_str, fuzzy=True, default=datetime(2025, 1, 1))
    return dt.strftime("%Y-%m-%d %H:%M")
