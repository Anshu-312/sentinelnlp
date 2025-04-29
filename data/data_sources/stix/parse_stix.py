# Script to parse and process STIX data import os
import json
import os

def parse_stix_bundle(raw_file):
    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_objects = []

    # STIX Bundles have "objects" list
    for obj in data.get("objects", []):
        parsed_obj = {
            "id": obj.get("id", ""),
            "type": obj.get("type", ""),
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
            "created": obj.get("created", ""),
            "modified": obj.get("modified", ""),
            "labels": obj.get("labels", []),
            "external_references": [],
            "kill_chain_phases": []
        }

        # External references (IDs, links)
        for ref in obj.get("external_references", []):
            parsed_obj["external_references"].append({
                "source_name": ref.get("source_name", ""),
                "external_id": ref.get("external_id", ""),
                "url": ref.get("url", "")
            })

        # Kill chain phases
        for phase in obj.get("kill_chain_phases", []):
            parsed_obj["kill_chain_phases"].append(phase.get("phase_name", ""))

        parsed_objects.append(parsed_obj)

    return parsed_objects

def save_parsed_stix(processed_file, parsed_objects):
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(parsed_objects, f, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(
        base_dir,
        "raw",
        "cti-stix2-json-schemas-master",
        "cti-stix2-json-schemas-master",
        "examples",
        "threat-reports",
        "apt1.json"
    )
    processed_file = os.path.join(base_dir, "processed", "parsed_stix_objects.json")
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)

    parsed_stix = parse_stix_bundle(raw_file)
    save_parsed_stix(processed_file, parsed_stix)
    print(f"Parsed {len(parsed_stix)} STIX objects!")
