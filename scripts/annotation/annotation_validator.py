import json
import os

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Load ontology
with open(os.path.join(PROJECT_ROOT, "ontology/entities.json")) as f:
    VALID_ENTITIES = set(et["name"] for et in json.load(f)["entity_types"])

with open(os.path.join(PROJECT_ROOT, "ontology/relations.json")) as f:
    RELATION_DEFS = json.load(f)["relation_types"]
RELATION_LOOKUP = {
    r["name"]: (set(r["source"]), set(r["target"])) for r in RELATION_DEFS
}

def validate_annotation(file_path):
    with open(file_path) as f:
        data = json.load(f)

    errors = []

    for idx, entry in enumerate(data):
        text = entry["text"]
        entities = entry["entities"]
        relations = entry.get("relations", [])

        for e in entities:
            if e["label"] not in VALID_ENTITIES:
                errors.append(f"[Entry {idx}] Invalid entity label: {e['label']}")
            if not (0 <= e["start"] < e["end"] <= len(text)):
                errors.append(f"[Entry {idx}] Invalid entity span: {e}")

        for r in relations:
            try:
                src = entities[r["source"]]
                tgt = entities[r["target"]]
                rel_type = r["type"]
                if rel_type not in RELATION_LOOKUP:
                    errors.append(f"[Entry {idx}] Invalid relation type: {rel_type}")
                else:
                    src_valid = src["label"] in RELATION_LOOKUP[rel_type][0]
                    tgt_valid = tgt["label"] in RELATION_LOOKUP[rel_type][1]
                    if not (src_valid and tgt_valid):
                        errors.append(f"[Entry {idx}] Invalid relation structure: {rel_type} from {src['label']} to {tgt['label']}")
            except IndexError:
                errors.append(f"[Entry {idx}] Relation points to invalid entity index")

    if not errors:
        print("✅ All annotations valid.")
    else:
        print("❌ Annotation errors found:")
        for e in errors:
            print(" -", e)

if __name__ == "__main__":
    input_file = os.path.join(PROJECT_ROOT, "data/annotated/auto_annotated.json")
    validate_annotation(input_file)
