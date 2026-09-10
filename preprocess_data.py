"""
Preprocess raw datasets into a unified, filtered question bank.
Streams the large HR JSON file instead of loading it all into memory.
"""

import json
import csv
import ijson  # streaming JSON parser for the large file

TARGET_ROLES = [
    "Software Engineer", "Data Scientist", "Product Manager",
    "DevOps Engineer", "QA Analyst", "UX Designer",
    "Marketing Associate", "HR Specialist"
]
MAX_PER_ROLE = 150  # cap so the working dataset stays small and fast

def preprocess_hr_data(input_path="data/hr_interview_questions_dataset.json",
                        output_path="data/unified_questions.json"):
    role_counts = {role: 0 for role in TARGET_ROLES}
    unified = []

    print("Streaming HR dataset (this may take a minute for a 1.2GB file)...")
    with open(input_path, "r", encoding="cp1252", errors="replace") as f:
        for entry in ijson.items(f, "item"):
            role = entry.get("role", "")
            if role in TARGET_ROLES and role_counts[role] < MAX_PER_ROLE:
                category = entry.get("category", "").lower()
                q_type = "behavioral" if entry.get("source_type") == "Behavioral" else "hr"
                unified.append({
                    "question": entry.get("question", ""),
                    "type": q_type,
                    "role": role,
                    "experience": entry.get("experience", "any"),
                    "difficulty": entry.get("difficulty", "Medium"),
                    "ideal_answer": entry.get("ideal_answer", "")
                })
                role_counts[role] += 1

            if sum(role_counts.values()) >= MAX_PER_ROLE * len(TARGET_ROLES):
                break  # stop early once we have enough

    print(f"Collected {len(unified)} HR/behavioral questions:", role_counts)
    return unified

def preprocess_software_data(input_path="data/Software Questions.csv"):
    unified = []
    with open(input_path, "r", encoding="cp1252", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unified.append({
                "question": row["Question"],
                "type": "technical",
                "role": "Software Engineer",
                "experience": "any",
                "difficulty": row.get("Difficulty", "Medium"),
                "ideal_answer": row.get("Answer", "")
            })
    print(f"Collected {len(unified)} technical questions")
    return unified

if __name__ == "__main__":
    hr_data = preprocess_hr_data()
    sw_data = preprocess_software_data()
    combined = hr_data + sw_data

    with open("data/unified_questions.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved {len(combined)} total questions to data/unified_questions.json")
