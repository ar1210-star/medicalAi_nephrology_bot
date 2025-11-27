# app/tools/patient_db.py

import json
import os
from typing import List, Dict, Any

# In-memory cache
patients:List[Dict] = []


def load_patients() -> List[Dict[str, Any]]:
    """Load patient data from JSON file."""
    global patients

    if patients:  # already loaded
        return patients

    fp = "data/patients.json" # retriving the file

    if not os.path.exists(fp):
        raise FileNotFoundError("Patient data file not found.") #throwing error if the file is not foundes

    with open(fp, "r", encoding="utf-8") as f:
        patients = json.load(f)# loading json file into list

    return patients


def get_all_patients() -> List[Dict]:
    """Return all patient records."""
    return load_patients()


def find_patient_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Find patients by name.
    """
    patients = load_patients()

    # normalize spaces and lower-case
    target = " ".join(name.strip().split()).lower()

    # 1) exact matches
    exact_matches: List[Dict[str, Any]] = [
        p
        for p in patients
        if " ".join(p.get("patient_name", "").strip().split()).lower() == target
    ]
    if exact_matches:
        return exact_matches

    # 2) fallback: substring matches
    substring_matches: List[Dict[str, Any]] = [
        p
        for p in patients
        if target in p.get("patient_name", "").lower()
    ]
    return substring_matches
