import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---- Configurations ----
REAL_ANNOTATION = Path(PROJECT_ROOT) / "data/annotated/auto_annotated.json"
SYNTHETIC_ANNOTATION = Path(PROJECT_ROOT) / "data/annotated/synthetic_annotated.json"
NER_OUT_DIR = Path(PROJECT_ROOT) / "data/processed/ner"
RE_OUT_DIR = Path(PROJECT_ROOT) / "data/processed/re"

# Create output directories
NER_OUT_DIR.mkdir(parents=True, exist_ok=True)
RE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load Real + Synthetic Data ----
print(f"Loading annotations from:")
print(f"  ➤ Real data: {REAL_ANNOTATION}")
print(f"  ➤ Synthetic data: {SYNTHETIC_ANNOTATION}")

with open(REAL_ANNOTATION) as f:
    real_data = json.load(f)

with open(SYNTHETIC_ANNOTATION) as f:
    synthetic_data = json.load(f)

combined_data = real_data + synthetic_data
print(f"🔗 Merged total {len(combined_data)} annotated samples (Real: {len(real_data)}, Synthetic: {len(synthetic_data)})")

# ---- Convert to NER Format ----
ner_data = []
for entry in combined_data:
    text = entry["text"]
    spans = [(ent["start"], ent["end"], ent["label"]) for ent in entry["entities"]]
    ner_data.append((text, {"entities": spans}))

# ---- Convert to RE Format ----
re_data = []
for entry in combined_data:
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

# ---- Train-Test Split ----
ner_train, ner_test = train_test_split(ner_data, test_size=0.2, random_state=42)
re_train, re_test = train_test_split(re_data, test_size=0.2, random_state=42)

# ---- Save Outputs ----
print("\nSaving processed datasets:")
print(f"  ➤ NER data directory: {NER_OUT_DIR}")
print(f"  ➤ RE data directory: {RE_OUT_DIR}")

with open(NER_OUT_DIR / "train.json", "w") as f:
    json.dump(ner_train, f, indent=2)
with open(NER_OUT_DIR / "test.json", "w") as f:
    json.dump(ner_test, f, indent=2)

with open(RE_OUT_DIR / "train.json", "w") as f:
    json.dump(re_train, f, indent=2)
with open(RE_OUT_DIR / "test.json", "w") as f:
    json.dump(re_test, f, indent=2)

print("\n✅ Dataset merge and conversion complete:")
print(f"  ➤ NER: {len(ner_train)} train / {len(ner_test)} test samples")
print(f"  ➤ RE : {len(re_train)} train / {len(re_test)} test samples")
