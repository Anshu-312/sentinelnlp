import os
import json

def extract_mitre_attack_objects(raw_file):
    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_items = []
    for obj in data.get("objects", []):
        if obj.get("type") in ["attack-pattern", "malware", "intrusion-set", "tool"]:
            item = {
                "id": obj.get("external_references", [{}])[0].get("external_id", ""),
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "type": obj.get("type", ""),
                "platforms": obj.get("x_mitre_platforms", []),
                "kill_chain_phases": [kcp["phase_name"] for kcp in obj.get("kill_chain_phases", [])]
            }
            extracted_items.append(item)
    
    return extracted_items

def save_extracted_items(processed_path, items):
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(base_dir, "raw", "cti-master", "cti-master", "enterprise-attack", "enterprise-attack.json")
    processed_file = os.path.join(base_dir, "processed", "enterprise_attack_cleaned.json")
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)

    mitre_items = extract_mitre_attack_objects(raw_file)
    save_extracted_items(processed_file, mitre_items)
    print(f"Extracted {len(mitre_items)} items from MITRE ATT&CK!")
