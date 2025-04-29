# Script to parse and process OTRF data import os
import os
import pandas as pd
import json

def parse_otrf_csv(raw_file):
    df = pd.read_csv(raw_file)

    parsed_entries = []
    for _, row in df.iterrows():
        entry = {
            "creation_date": row.get("Creation Date", ""),
            "id": row.get("Id", ""),
            "title": row.get("Title", ""),
            "tactics": row.get("Tactics", ""),
            "collaborators": row.get("Collaborators", "")
        }
        parsed_entries.append(entry)
    
    return parsed_entries

def save_parsed_entries(processed_file, entries):
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(
        base_dir,
        "raw",
        "ThreatHunter-Playbook-master",
        "ThreatHunter-Playbook-master",
        "docs",
        "hunts",
        "windows",
        "analytic_summary.csv"
    )
    processed_file = os.path.join(base_dir, "processed", "otrf_analytic_summary.json")
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)

    otrf_entries = parse_otrf_csv(raw_file)
    save_parsed_entries(processed_file, otrf_entries)
    print(f"Parsed {len(otrf_entries)} entries from OTRF analytic summary!")
