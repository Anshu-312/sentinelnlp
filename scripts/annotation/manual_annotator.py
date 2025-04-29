import json
import re
import spacy
import os

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load ontology
with open(os.path.join(PROJECT_ROOT, "ontology/entities.json")) as f:
    ENTITY_TYPES = set(et["name"] for et in json.load(f)["entity_types"])

with open(os.path.join(PROJECT_ROOT, "ontology/relations.json")) as f:
    RELATION_DEFS = json.load(f)["relation_types"]

# Simple patterns for demonstration
PATTERNS = {
    "THREAT_ACTOR": [r"\bAPT\d{2,}\b", r"Lazarus Group", r"Fancy Bear"],
    "MALWARE": [r"\b(Emotet|WannaCry|TrickBot|NotPetya)\b"],
    "TOOL": [r"\b(Mimikatz|Cobalt Strike|Empire)\b"],
    "VULNERABILITY": [r"\bCVE-\d{4}-\d{4,7}\b"],
    "TACTIC": [r"Initial Access", r"Lateral Movement"],
    "TECHNIQUE": [r"Spearphishing Link", r"Process Injection", r"DLL Sideloading"],
    "TARGET": [r"Microsoft", r"government", r"organization", r"bank"],
    "INFRASTRUCTURE": [r"\b\d{1,3}(?:\.\d{1,3}){3}\b", r"\b[a-zA-Z0-9.-]+\.(com|net|org)\b"]
}


def annotate_text(text):
    doc = nlp(text)
    entities = []
    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": label
                })

    # Build index map for relation linking
    rels = []
    for i, ent1 in enumerate(entities):
        for j, ent2 in enumerate(entities):
            if i == j:
                continue
            for r in RELATION_DEFS:
                if ent1["label"] in r["source"] and ent2["label"] in r["target"]:
                    rels.append({
                        "source": i,
                        "target": j,
                        "type": r["name"]
                    })

    return {
        "text": text,
        "entities": entities,
        "relations": rels
    }


if __name__ == "__main__":
    sample_inputs = [
        "APT29 used Mimikatz to escalate privileges via DLL Sideloading.",
        "The malware Emotet exploited CVE-2021-34527 to gain access to Microsoft systems.",
        "Lazarus Group deployed TrickBot on IP 123.45.67.89 targeting government entities."
    ]

    annotated = [annotate_text(txt) for txt in sample_inputs]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(PROJECT_ROOT, "data/annotated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotations
    output_file = os.path.join(output_dir, "auto_annotated.json")
    with open(output_file, "w") as f:
        json.dump(annotated, f, indent=2)

    print(f"✅ Auto annotation completed and saved to {output_file}")
