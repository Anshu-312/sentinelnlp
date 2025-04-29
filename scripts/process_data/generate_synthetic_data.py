import random
import json
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_FILE = Path(PROJECT_ROOT) / "data/annotated/synthetic_annotated.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Sample ontology-driven entity pools
ENTITIES = {
    "THREAT_ACTOR": ["APT28", "Lazarus Group", "APT29"],
    "MALWARE": ["Emotet", "TrickBot", "RansomEXX"],
    "TOOL": ["Mimikatz", "Cobalt Strike", "PowerSploit"],
    "VULNERABILITY": ["CVE-2023-12345", "CVE-2021-34527"],
    "TECHNIQUE": ["DLL Sideloading", "Phishing", "Command Injection"],
    "TARGET": ["government entities", "Microsoft systems", "banks"],
    "INFRASTRUCTURE": ["192.168.1.10", "203.0.113.5", "C2 server"],
}

# Relation templates with consistent naming
TEMPLATES = [
    {
        "template": "{threat_actor} used {tool} to perform {technique}.",
        "entities": {
            "threat_actor": "THREAT_ACTOR",
            "tool": "TOOL",
            "technique": "TECHNIQUE"
        },
        "relations": [
            ("THREAT_ACTOR", "TOOL", "uses"),
            ("THREAT_ACTOR", "TECHNIQUE", "executes"),
            ("TOOL", "TECHNIQUE", "facilitates")
        ]
    },
    {
        "template": "{malware} exploited {vulnerability} to infect {target}.",
        "entities": {
            "malware": "MALWARE",
            "vulnerability": "VULNERABILITY",
            "target": "TARGET"
        },
        "relations": [
            ("MALWARE", "VULNERABILITY", "exploits"),
            ("MALWARE", "TARGET", "targets")
        ]
    },
    {
        "template": "{threat_actor} deployed {malware} on {infrastructure} targeting {target}.",
        "entities": {
            "threat_actor": "THREAT_ACTOR",
            "malware": "MALWARE",
            "infrastructure": "INFRASTRUCTURE",
            "target": "TARGET"
        },
        "relations": [
            ("THREAT_ACTOR", "MALWARE", "uses"),
            ("THREAT_ACTOR", "TARGET", "targets"),
            ("INFRASTRUCTURE", "MALWARE", "hosts")
        ]
    }
]

def generate_annotated_sample():
    template = random.choice(TEMPLATES)
    # Generate values for each template slot
    entity_values = {}
    for template_key, entity_type in template["entities"].items():
        entity_values[template_key] = random.choice(ENTITIES[entity_type])
    
    # Format the sentence
    sentence = template["template"].format(**entity_values)

    # Track entities and their positions
    entities = []
    used = {}
    for template_key, entity_type in template["entities"].items():
        value = entity_values[template_key]
        start = sentence.find(value)
        if start == -1:
            continue  # skip if something is wrong
        end = start + len(value)
        if value in used:
            # Avoid duplicates
            start = sentence.find(value, used[value] + 1)
            end = start + len(value)
        used[value] = start
        entities.append({"start": start, "end": end, "label": entity_type})

    # Generate relations
    relations = []
    for src_type, tgt_type, rel in template["relations"]:
        src_idx = next((i for i, ent in enumerate(entities) if ent["label"] == src_type), None)
        tgt_idx = next((i for i, ent in enumerate(entities) if ent["label"] == tgt_type), None)
        if src_idx is not None and tgt_idx is not None:
            relations.append({"source": src_idx, "target": tgt_idx, "type": rel})

    return {
        "text": sentence,
        "entities": entities,
        "relations": relations
    }

# Generate synthetic samples
synthetic_data = [generate_annotated_sample() for _ in range(100)]

# Save the generated data
with open(OUTPUT_FILE, "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"✅ Generated {len(synthetic_data)} synthetic samples")
print(f"   Saved to: {OUTPUT_FILE}")
