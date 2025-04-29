import json
import random
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define paths relative to project root
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, "data/annotated/auto_annotated.json")
NER_OUT_DIR = Path(PROJECT_ROOT) / "data/processed/ner"
RE_OUT_DIR = Path(PROJECT_ROOT) / "data/processed/re"

# Create output directories
NER_OUT_DIR.mkdir(parents=True, exist_ok=True)
RE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(ANNOTATION_FILE) as f:
    data = json.load(f)

# ---------- 1. NER Dataset Conversion ----------
ner_data = []
for entry in data:
    text = entry["text"]
    spans = [(ent["start"], ent["end"], ent["label"]) for ent in entry["entities"]]
    ner_data.append((text, {"entities": spans}))

# ---------- 2. RE Dataset Conversion ----------
re_data = []
for entry in data:
    text = entry["text"]
    entities = entry["entities"]
    for rel in entry.get("relations", []):
        src = entities[rel["source"]]
        tgt = entities[rel["target"]]
        re_data.append({
            "text": text,
            "entity_1": {"text": text[src["start"]:src["end"]], "label": src["label"]},
            "entity_2": {"text": text[tgt["start"]:tgt["end"]], "label": tgt["label"]},
            "relation": rel["type"]
        })

# ---------- 3. Train-Test Split ----------
ner_train, ner_test = train_test_split(ner_data, test_size=0.2, random_state=42)
re_train, re_test = train_test_split(re_data, test_size=0.2, random_state=42)

# ---------- 4. Save Datasets ----------
with open(NER_OUT_DIR / "train.json", "w") as f:
    json.dump(ner_train, f, indent=2)
with open(NER_OUT_DIR / "test.json", "w") as f:
    json.dump(ner_test, f, indent=2)

with open(RE_OUT_DIR / "train.json", "w") as f:
    json.dump(re_train, f, indent=2)
with open(RE_OUT_DIR / "test.json", "w") as f:
    json.dump(re_test, f, indent=2)

print(f"✅ NER & RE datasets prepared and split into train/test sets.")
print(f"   NER data saved to: {NER_OUT_DIR}")
print(f"   RE data saved to: {RE_OUT_DIR}")
